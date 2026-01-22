from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import errno
import io
import os
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import files
import six
@six.add_metaclass(abc.ABCMeta)
class DeclarativeClientBase(object):
    """KRM Yaml Export based Declarative Client."""

    @property
    @abc.abstractmethod
    def binary_name(self):
        pass

    @property
    @abc.abstractmethod
    def binary_prompt(self):
        pass

    @abc.abstractmethod
    def BulkExport(self, args):
        pass

    @abc.abstractmethod
    def ExportAll(self, args):
        pass

    def __init__(self, gcp_account=None, impersonated=False):
        from googlecloudsdk.command_lib.util.anthos import binary_operations as bin_ops
        if not gcp_account:
            gcp_account = properties.VALUES.core.account.Get()
        try:
            self._export_service = bin_ops.CheckForInstalledBinary(self.binary_name)
        except bin_ops.MissingExecutableException:
            self._export_service = bin_ops.InstallBinaryNoOverrides(self.binary_name, prompt=self.binary_prompt)
        self._use_account_impersonation = impersonated
        self._account = gcp_account

    def _GetToken(self):
        from googlecloudsdk.core.credentials import store
        try:
            return store.GetFreshAccessToken(account=self._account, allow_account_impersonation=self._use_account_impersonation)
        except Exception as e:
            raise ClientException('Error Configuring KCC Client: [{}]'.format(e))

    def _OutputToFileOrDir(self, path):
        if path.strip() == '-':
            return False
        return True

    def _TryCreateOutputPath(self, path):
        """Try to create output directory if it doesnt exists."""
        directory = os.path.abspath(path.strip())
        try:
            if os.path.isdir(directory) and files.HasWriteAccessInDir(directory):
                return
            if files.HasWriteAccessInDir(os.path.dirname(directory)):
                console_io.PromptContinue('Path {} does not exists. Do you want to create it?'.format(path), default=True, cancel_on_no=True, cancel_string='Export aborted. No files written.')
                files.MakeDir(path)
            else:
                raise OSError(errno.EACCES)
        except ValueError:
            raise ExportPathException('Can not export to path. [{}] is not a directory.'.format(path))
        except OSError:
            raise ExportPathException('Can not export to path [{}]. Ensure that enclosing path exists and is writeable.'.format(path))

    def _ParseResourceTypes(self, args):
        return getattr(args, 'resource_types', None) or self._ParseKindTypesFileData(getattr(args, 'resource_types_file', None))

    def _GetBinaryExportCommand(self, args, command_name, resource_uri=None, skip_parent=False, skip_filter=False):
        """Constructs and returns a list representing the binary export command."""
        cmd = [self._export_service, '--oauth2-token', self._GetToken(), command_name]
        if command_name == 'export':
            if not resource_uri:
                raise ClientException('`_GetBinaryExportCommand` requires a resource uri for export commands.')
            cmd.extend([resource_uri])
        if command_name == 'bulk-export':
            cmd.extend(['--on-error', getattr(args, 'on_error', 'ignore')])
            if not skip_parent:
                if args.IsSpecified('organization'):
                    cmd.extend(['--organization', args.organization])
                elif args.IsSpecified('folder'):
                    cmd.extend(['--folder', args.folder])
                else:
                    project = args.project or properties.VALUES.core.project.GetOrFail()
                    cmd.extend(['--project', project])
            if not skip_filter:
                if args.IsSpecified('resource_types') or args.IsSpecified('resource_types_file'):
                    cmd.extend(['--resource-types', self._ParseResourceTypes(args)])
        if getattr(args, 'storage_path', None):
            cmd.extend(['--storage-key', args.storage_path])
        if getattr(args, 'resource_format', None):
            cmd.extend(['--resource-format', _NormalizeResourceFormat(args.resource_format)])
            if args.resource_format == 'terraform':
                cmd.extend(['--iam-format', 'none'])
        if self._OutputToFileOrDir(args.path):
            cmd.extend(['--output', args.path])
        return cmd

    def Export(self, args, resource_uri):
        """Exports a single resource's configuration file."""
        normalized_resource_uri = _NormalizeUri(resource_uri)
        with progress_tracker.ProgressTracker(message='Exporting resources', aborted_message='Aborted Export.'):
            cmd = self._GetBinaryExportCommand(args=args, command_name='export', resource_uri=normalized_resource_uri)
            exit_code, output_value, error_value = _ExecuteBinary(cmd)
        if exit_code != 0:
            if 'resource not found' in error_value:
                raise ResourceNotFoundException('Could not fetch resource: \n - The resource [{}] does not exist.'.format(normalized_resource_uri))
            elif 'Error 403' in error_value:
                raise ClientException('Permission Denied during export. Please ensure resource API is enabled for resource [{}] and Cloud IAM permissions are set properly.'.format(resource_uri))
            else:
                raise ClientException('Error executing export:: [{}]'.format(error_value))
        if not self._OutputToFileOrDir(args.path):
            log.out.Print(output_value)
        log.status.Print('Exported successfully.')
        return exit_code

    def _CallBulkExport(self, cmd, args, asset_list_input=None):
        """Execute actual bulk-export command on config-connector binary."""
        if self._OutputToFileOrDir(args.path):
            self._TryCreateOutputPath(args.path)
            preexisting_file_count = sum([len(files_in_dir) for r, d, files_in_dir in os.walk(args.path)])
            with progress_tracker.ProgressTracker(message='Exporting resource configurations to [{}]'.format(args.path), aborted_message='Aborted Export.'):
                exit_code, _, error_value = _ExecuteBinary(cmd=cmd, in_str=asset_list_input)
            if exit_code != 0:
                if 'Error 403' in error_value:
                    msg = 'Permission denied during export. Please ensure the Cloud Asset Inventory API is enabled.'
                    if args.storage_path:
                        msg += ' Also check that Cloud IAM permissions are set for `--storage-path` [{}]'.format(args.storage_path)
                    raise ClientException(msg)
                else:
                    raise ClientException('Error executing export:: [{}]'.format(error_value))
            else:
                _BulkExportPostStatus(preexisting_file_count, args.path)
            return exit_code
        else:
            log.status.write('Exporting resource configurations to stdout...\n')
            return _ExecuteBinaryWithStreaming(cmd=cmd, in_str=asset_list_input)

    def _CallPrintResources(self, output_format='table'):
        """Calls `print-resources` on the underlying binary."""
        cmd = [self._export_service, 'print-resources', '--output-format', output_format]
        exit_code, output_value, error_value = _ExecuteBinary(cmd)
        if exit_code != 0:
            raise ClientException('Error occured while listing available resources: [{}]'.format(error_value))
        return output_value

    def ListResources(self, project=None, organization=None, folder=None):
        """List all exportable resources.

    If parent (e.g. project, organization or folder) is passed then only list
    the exportable resources for that parent.

    Args:
      project: string, project to list exportable resources for.
      organization: string, organization to list exportable resources for.
      folder: string, folder to list exportable resources for.

    Returns:
      supported resources formatted output listing exportable resources.

    """
        if not (project or organization or folder):
            yaml_obj_list = yaml.load(self._CallPrintResources(output_format='yaml'), round_trip=True)
            return yaml_obj_list
        if project:
            msg_sfx = ' for project [{}]'.format(project)
        elif organization:
            msg_sfx = ' for organization [{}]'.format(organization)
        else:
            msg_sfx = ' for folder [{}]'.format(folder)
        with progress_tracker.ProgressTracker(message='Listing exportable resource types' + msg_sfx, aborted_message='Aborted Export.'):
            supported_kinds = self.ListSupportedResourcesForParent(project=project, organization=organization, folder=folder)
            supported_kinds = [x.AsDict() for x in supported_kinds]
            return supported_kinds

    def ListSupportedResourcesForParent(self, project=None, organization=None, folder=None):
        """List all exportable resource types for a given project, org or folder."""
        if not (project or organization or folder):
            raise ClientException('At least one of project, organization or folder must be specified for this operation')
        name_translator = resource_name_translator.ResourceNameTranslator()
        asset_list_data = GetAssetInventoryListInput(folder=folder, org=organization, project=project)
        asset_types = set([x.replace('"', '') for x in _ASSET_TYPE_REGEX.findall(asset_list_data)])
        exportable_kinds = []
        for asset in asset_types:
            try:
                meta_resource = name_translator.get_resource(asset_inventory_type=asset)
                gvk = KrmGroupValueKind(kind=meta_resource.krm_kind.krm_kind, group=meta_resource.krm_kind.krm_group + _KRM_GROUP_SUFFIX, bulk_export_supported=meta_resource.resource_data.support_bulk_export, export_supported=meta_resource.resource_data.support_single_export, iam_supported=meta_resource.resource_data.support_iam)
                exportable_kinds.append(gvk)
            except resource_name_translator.ResourceIdentifierNotFoundError:
                continue
        return sorted(exportable_kinds, key=lambda x: x.kind)

    def ApplyConfig(self, input_path, try_resolve_refs=False):
        """Call apply from config-connector binary.

    Applys the KRM config file specified by `path`, creating or updating the
    related GCP resources. If `try_resolve_refs` is supplied, then command will
    attempt to resolve the references to related resources in config,
    creating a directed graph of related resources and apply them in order.

    Args:
      input_path: string, KRM config file to apply.
      try_resolve_refs: boolean, if true attempt to resolve the references to
      related resources in config.

    Returns:
      Yaml Object representing the updated state of the resource if successful.

    Raises:
      ApplyException: if an error occurs applying config.
      ApplyPathException: if an error occurs reading file path.
    """
        del try_resolve_refs
        if not input_path or not input_path.strip() or (not os.path.isfile(input_path)):
            raise ApplyPathException('Resource file path [{}] not found.'.format(input_path))
        cmd = [self._export_service, 'apply', '-i', input_path, '--oauth2-token', self._GetToken()]
        exit_code, output_value, error_value = _ExecuteBinary(cmd)
        if exit_code != 0:
            raise ApplyException('Error occured while applying configuration path [{}]: [{}]'.format(input_path, error_value))
        return yaml.load(output_value)