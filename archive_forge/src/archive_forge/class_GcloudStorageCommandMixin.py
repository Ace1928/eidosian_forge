from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
import re
import subprocess
from boto import config
from gslib import exception
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
class GcloudStorageCommandMixin(object):
    """Provides gcloud storage translation functionality.

  The command.Command class must inherit this class in order to support
  converting the gsutil command to it's gcloud storage equivalent.
  """
    gcloud_storage_map = None

    def __init__(self):
        self._translated_gcloud_storage_command = None
        self._translated_env_variables = None

    def _get_gcloud_storage_args(self, sub_opts, gsutil_args, gcloud_storage_map):
        if gcloud_storage_map is None:
            raise exception.GcloudStorageTranslationError('Command "{}" cannot be translated to gcloud storage because the translation mapping is missing.'.format(self.command_name))
        args = []
        if isinstance(gcloud_storage_map.gcloud_command, list):
            args.extend(gcloud_storage_map.gcloud_command)
        elif isinstance(gcloud_storage_map.gcloud_command, dict):
            if gcloud_storage_map.flag_map:
                raise ValueError('Flags mapping should not be present at the top-level command if a sub-command is used. Command: {}.'.format(self.command_name))
            sub_command = gsutil_args[0]
            sub_opts, parsed_args = self.ParseSubOpts(args=gsutil_args[1:], should_update_sub_opts_and_args=False)
            return self._get_gcloud_storage_args(sub_opts, parsed_args, gcloud_storage_map.gcloud_command.get(sub_command))
        else:
            raise ValueError('Incorrect mapping found for "{}" command'.format(self.command_name))
        if sub_opts:
            for option, value in sub_opts:
                if option not in gcloud_storage_map.flag_map:
                    raise exception.GcloudStorageTranslationError('Command option "{}" cannot be translated to gcloud storage'.format(option))
                else:
                    args.append(option)
                    if value != '':
                        args.append(value)
        return _convert_args_to_gcloud_values(args + gsutil_args, gcloud_storage_map)

    def _translate_top_level_flags(self):
        """Translates gsutil's top level flags.

    Gsutil specifies the headers (-h) and boto config (-o) as top level flags
    as well, but we handle those separately.

    Returns:
      A tuple. The first item is a list of top level flags that can be appended
        to the gcloud storage command. The second item is a dict of environment
        variables that can be set for the gcloud storage command execution.
    """
        top_level_flags = []
        env_variables = {'CLOUDSDK_METRICS_ENVIRONMENT': 'gsutil_shim', 'CLOUDSDK_STORAGE_RUN_BY_GSUTIL_SHIM': 'True'}
        if self.debug >= 3:
            top_level_flags.extend(['--verbosity', 'debug'])
        if self.debug == 4:
            top_level_flags.append('--log-http')
        if self.quiet_mode:
            top_level_flags.append('--no-user-output-enabled')
        if self.user_project:
            top_level_flags.append('--billing-project=' + self.user_project)
        if self.trace_token:
            top_level_flags.append('--trace-token=' + self.trace_token)
        if constants.IMPERSONATE_SERVICE_ACCOUNT:
            top_level_flags.append('--impersonate-service-account=' + constants.IMPERSONATE_SERVICE_ACCOUNT)
        should_use_rsync_override = self.command_name == 'rsync' and (not (config.get('GSUtil', 'parallel_process_count') == '1' and config.get('GSUtil', 'thread_process_count') == '1'))
        if not (self.parallel_operations or should_use_rsync_override):
            env_variables['CLOUDSDK_STORAGE_THREAD_COUNT'] = '1'
            env_variables['CLOUDSDK_STORAGE_PROCESS_COUNT'] = '1'
        return (top_level_flags, env_variables)

    def _translate_headers(self, headers=None, unset=False):
        """Translates gsutil headers to equivalent gcloud storage flags.

    Args:
      headers (dict|None): If absent, extracts headers from command instance.
      unset (bool): Yield metadata clear flags instead of setter flags.

    Returns:
      List[str]: Translated flags for gcloud.

    Raises:
      GcloudStorageTranslationError: Could not translate flag.
    """
        flags = []
        headers_to_translate = headers if headers is not None else self.headers
        additional_headers = []
        for raw_header_key, header_value in headers_to_translate.items():
            lowercase_header_key = raw_header_key.lower()
            if lowercase_header_key == 'x-goog-api-version':
                continue
            flag = get_flag_from_header(raw_header_key, header_value, unset=unset)
            if self.command_name in COMMANDS_SUPPORTING_ALL_HEADERS:
                if flag:
                    flags.append(flag)
            elif self.command_name in PRECONDITONS_ONLY_SUPPORTED_COMMANDS and lowercase_header_key in PRECONDITIONS_HEADERS:
                flags.append(flag)
            if not flag:
                self.logger.warn('Header {}:{} cannot be translated to a gcloud storage equivalent flag. It is being treated as an arbitrary request header.'.format(raw_header_key, header_value))
                additional_headers.append('{}={}'.format(raw_header_key, header_value))
        if additional_headers:
            flags.append('--additional-headers=' + ','.join(additional_headers))
        return flags

    def _translate_boto_config(self):
        """Translates boto config options to gcloud storage properties.

    Returns:
      A tuple where first element is a list of flags and the second element is
      a dict representing the env variables that can be set to set the
      gcloud storage properties.
    """
        flags = []
        env_vars = {}
        gcs_json_endpoint = _get_gcs_json_endpoint_from_boto_config(config)
        if gcs_json_endpoint:
            env_vars['CLOUDSDK_API_ENDPOINT_OVERRIDES_STORAGE'] = gcs_json_endpoint
        s3_endpoint = _get_s3_endpoint_from_boto_config(config)
        if s3_endpoint:
            env_vars['CLOUDSDK_STORAGE_S3_ENDPOINT_URL'] = s3_endpoint
        decryption_keys = []
        for section_name, section in config.items():
            for key, value in section.items():
                if key == 'encryption_key' and self.command_name in ENCRYPTION_SUPPORTED_COMMANDS:
                    flags.append('--encryption-key=' + value)
                elif DECRYPTION_KEY_REGEX.match(key) and self.command_name in ENCRYPTION_SUPPORTED_COMMANDS:
                    decryption_keys.append(value)
                elif key == 'content_language' and self.command_name in COMMANDS_SUPPORTING_ALL_HEADERS:
                    flags.append('--content-language=' + value)
                elif key in _REQUIRED_BOTO_CONFIG_NOT_YET_SUPPORTED:
                    self.logger.error('The boto config field {}:{} cannot be translated to gcloud storage equivalent.'.format(section_name, key))
                elif key == 'https_validate_certificates' and (not value):
                    env_vars['CLOUDSDK_AUTH_DISABLE_SSL_VALIDATION'] = True
                elif key in ('gs_access_key_id', 'gs_secret_access_key') and (not boto_util.UsingGsHmac()):
                    self.logger.debug('The boto config field {}:{} skipped translation to the gcloud storage equivalent as it would have been unused in gsutil.'.format(section_name, key))
                else:
                    env_var = _BOTO_CONFIG_MAP.get(section_name, {}).get(key, None)
                    if env_var is not None:
                        env_vars[env_var] = value
        if decryption_keys:
            flags.append('--decryption-keys=' + ','.join(decryption_keys))
        return (flags, env_vars)

    def get_gcloud_storage_args(self, gcloud_storage_map=None):
        """Translates the gsutil command flags to gcloud storage flags.

    It uses the command_spec.gcloud_storage_map field that provides the
    translation mapping for all the flags.

    Args:
      gcloud_storage_map (GcloudStorageMap|None): Command surface may pass a
        custom translation map instead of using the default class constant.
        Useful for when translations change based on conditional logic.


    Returns:
      A list of all the options and arguments that can be used with the
        equivalent gcloud storage command.
    Raises:
      GcloudStorageTranslationError: If a flag or command cannot be translated.
      ValueError: If there is any issue with the mapping provided by
        GcloudStorageMap.
    """
        return self._get_gcloud_storage_args(self.sub_opts, self.args, gcloud_storage_map or self.gcloud_storage_map)

    def _print_gcloud_storage_command_info(self, gcloud_command, env_variables, dry_run=False):
        logger_func = self.logger.info if dry_run else self.logger.debug
        logger_func('Gcloud Storage Command: {}'.format(' '.join(gcloud_command)))
        if env_variables:
            logger_func('Environment variables for Gcloud Storage:')
            for k, v in env_variables.items():
                logger_func('%s=%s', k, v)

    def _get_full_gcloud_storage_execution_information(self, args):
        top_level_flags, env_variables = self._translate_top_level_flags()
        header_flags = self._translate_headers()
        flags_from_boto, env_vars_from_boto = self._translate_boto_config()
        env_variables.update(env_vars_from_boto)
        gcloud_binary_path = _get_validated_gcloud_binary_path()
        gcloud_storage_command = [gcloud_binary_path] + args + top_level_flags + header_flags + flags_from_boto
        return (env_variables, gcloud_storage_command)

    def translate_to_gcloud_storage_if_requested(self):
        """Translates the gsutil command to gcloud storage equivalent.

    The translated commands get stored at
    self._translated_gcloud_storage_command.
    This command also translate the boto config, which gets stored as a dict
    at self._translated_env_variables

    Returns:
      True if the command was successfully translated, else False.
    """
        if self.command_name == 'version' or self.command_name == 'test':
            return False
        use_gcloud_storage = config.getbool('GSUtil', 'use_gcloud_storage', False)
        try:
            hidden_shim_mode = HIDDEN_SHIM_MODE(config.get('GSUtil', 'hidden_shim_mode', 'none'))
        except ValueError:
            raise exception.CommandException('Invalid option specified for GSUtil:hidden_shim_mode config setting. Should be one of: {}'.format(' | '.join([x.value for x in HIDDEN_SHIM_MODE])))
        if use_gcloud_storage:
            try:
                env_variables, gcloud_storage_command = self._get_full_gcloud_storage_execution_information(self.get_gcloud_storage_args())
                if hidden_shim_mode == HIDDEN_SHIM_MODE.DRY_RUN:
                    self._print_gcloud_storage_command_info(gcloud_storage_command, env_variables, dry_run=True)
                elif not os.environ.get('CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL'):
                    raise exception.GcloudStorageTranslationError('Requested to use "gcloud storage" but gsutil is not using the same credentials as gcloud. You can make gsutil use the same credentials by running:\n{} config set pass_credentials_to_gsutil True'.format(_get_validated_gcloud_binary_path()))
                elif boto_util.UsingGsHmac() and ApiSelector.XML not in self.command_spec.gs_api_support:
                    raise CommandException('Requested to use "gcloud storage" with Cloud Storage XML API HMAC credentials but the "{}" command can only be used with the Cloud Storage JSON API.'.format(self.command_name))
                else:
                    self._print_gcloud_storage_command_info(gcloud_storage_command, env_variables)
                    self._translated_gcloud_storage_command = gcloud_storage_command
                    self._translated_env_variables = env_variables
                    return True
            except exception.GcloudStorageTranslationError as e:
                if hidden_shim_mode == HIDDEN_SHIM_MODE.NO_FALLBACK:
                    raise exception.CommandException(e)
                self.logger.error('Cannot translate gsutil command to gcloud storage. Going to run gsutil command. Error: %s', e)
        return False

    def _get_shim_command_environment_variables(self):
        subprocess_envs = os.environ.copy()
        subprocess_envs.update(self._translated_env_variables)
        return subprocess_envs

    def run_gcloud_storage(self):
        process = subprocess.run(self._translated_gcloud_storage_command, env=self._get_shim_command_environment_variables())
        return process.returncode