from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
class ComponentSnapshot(object):
    """Contains a state-of-the-world for existing components.

  A snapshot can be loaded from different sources.  It can be the latest that
  exists on the server or it can be constructed from local install state.
  Either way, it describes the components that are available, how they depend
  on each other, and other information about them like descriptions and version
  information.

  Attributes:
    revision: int, The global revision number for this snapshot.  If it was
      created from an InstallState, this will be -1 to indicate that it is
      potentially a composition of more than one snapshot.
    sdk_definition: schemas.SDKDefinition, The full definition for this
      component snapshot.
    url: str, The full URL of the file from which this snapshot was loaded.
      This could be a web address like http://internet.com/components.json or
      a local file path as a URL like file:///some/local/path/components.json.
      It may also be None if the data did not come from a file.
    components = dict from component id string to schemas.Component, All the
      Components in this snapshot.
  """
    ABSOLUTE_RE = re.compile('^\\w+://')

    @staticmethod
    def _GetAbsoluteURL(url, value):
        """Convert the potentially relative value into an absolute URL.

    Args:
      url: str, The URL of the component snapshot this value was found in.
      value: str, The value of the field to make absolute.  If it is already an
        absolute URL, it is returned as-is.  If it is relative, it's path
        is assumed to be relative to the component snapshot URL.

    Returns:
      str, The absolute URL.
    """
        if ComponentSnapshot.ABSOLUTE_RE.search(value):
            return value
        return os.path.dirname(url) + '/' + value

    @staticmethod
    def FromFile(snapshot_file):
        """Loads a snapshot from a local file.

    Args:
      snapshot_file: str, The path of the file to load.

    Returns:
      A ComponentSnapshot object
    """
        with files.FileReader(snapshot_file) as f:
            data = json.load(f)
        url = 'file://' + ('/' if not snapshot_file.startswith('/') else '') + snapshot_file.replace('\\', '/')
        return ComponentSnapshot._FromDictionary((data, url))

    @staticmethod
    def FromURLs(*urls, **kwargs):
        """Loads a snapshot from a series of URLs.

    Args:
      *urls: str, The URLs to the files to load.
      **kwargs: command_path: the command path to include in the User-Agent
        header if the URL is HTTP

    Returns:
      A ComponentSnapshot object.

    Raises:
      URLFetchError: If the URL cannot be fetched.
      TypeError: If an unexpected keyword argument is given.
    """
        current_function_name = ComponentSnapshot.FromURLs.__name__
        unexpected_args = set(kwargs) - set(['command_path'])
        if unexpected_args:
            raise TypeError("{0} got an unexpected keyword argument '{1}'".format(current_function_name, unexpected_args.pop()))
        command_path = kwargs.get('command_path', 'unknown')
        first = urls[0]
        data = [(ComponentSnapshot._DictFromURL(url, command_path, is_extra_repo=url != first), url) for url in urls]
        return ComponentSnapshot._FromDictionary(*data)

    @staticmethod
    def _DictFromURL(url, command_path, is_extra_repo=False):
        """Loads a json dictionary from a URL.

    Args:
      url: str, The URL to the file to load.
      command_path: the command path to include in the User-Agent header if the
        URL is HTTP
      is_extra_repo: bool, True if this is not the primary repository.

    Returns:
      A ComponentSnapshot object.

    Raises:
      URLFetchError: If the URL cannot be fetched.
    """
        extra_repo = url if is_extra_repo else None
        try:
            response = installers.MakeRequest(url, command_path)
        except requests.exceptions.HTTPError:
            log.debug('Could not fetch [{url}]'.format(url=url), exc_info=True)
            response = None
        if response is None:
            raise URLFetchError(extra_repo=extra_repo)
        code = response.status_code
        if code != requests.codes.ok:
            raise URLFetchError(code=code, extra_repo=extra_repo)
        try:
            data = json.loads(response.content.decode('utf-8'))
            return data
        except ValueError as e:
            log.debug('Failed to parse snapshot [{}]: {}'.format(url, e))
            raise MalformedSnapshotError()

    @staticmethod
    def FromInstallState(install_state):
        """Loads a snapshot from the local installation state.

    This creates a snapshot that may not have actually existed at any point in
    time.  It does, however, exactly reflect the current state of your local
    SDK.

    Args:
      install_state: install_state.InstallState, The InstallState object to load
        from.

    Returns:
      A ComponentSnapshot object.
    """
        installed = install_state.InstalledComponents()
        components = [manifest.ComponentDefinition() for manifest in installed.values()]
        sdk_definition = schemas.SDKDefinition(revision=-1, schema_version=None, release_notes_url=None, version=None, gcloud_rel_path=None, post_processing_command=None, components=components, notifications={})
        return ComponentSnapshot(sdk_definition)

    @staticmethod
    def _FromDictionary(*data):
        """Loads a snapshot from a dictionary representing the raw JSON data.

    Args:
      *data: ({}, str), A tuple of parsed JSON data and the URL it came from.

    Returns:
      A ComponentSnapshot object.

    Raises:
      IncompatibleSchemaVersionError: If the latest snapshot cannot be parsed
        by this code.
    """
        merged = None
        for json_dictionary, url in data:
            schema_version = schemas.SDKDefinition.SchemaVersion(json_dictionary)
            if url and schema_version.url:
                schema_version.url = ComponentSnapshot._GetAbsoluteURL(url, schema_version.url)
            if schema_version.version > config.INSTALLATION_CONFIG.snapshot_schema_version:
                raise IncompatibleSchemaVersionError(schema_version)
            sdk_def = schemas.SDKDefinition.FromDictionary(json_dictionary)
            if url:
                if sdk_def.schema_version.url:
                    sdk_def.schema_version.url = ComponentSnapshot._GetAbsoluteURL(url, sdk_def.schema_version.url)
                if sdk_def.release_notes_url:
                    sdk_def.release_notes_url = ComponentSnapshot._GetAbsoluteURL(url, sdk_def.release_notes_url)
                for c in sdk_def.components:
                    if not c.data or not c.data.source:
                        continue
                    c.data.source = ComponentSnapshot._GetAbsoluteURL(url, c.data.source)
            if not merged:
                merged = sdk_def
            else:
                merged.Merge(sdk_def)
        return ComponentSnapshot(merged)

    def __init__(self, sdk_definition):
        self.sdk_definition = sdk_definition
        self.revision = sdk_definition.revision
        self.version = sdk_definition.version
        self.components = dict(((c.id, c) for c in sdk_definition.components))
        deps = dict(((c.id, set(c.dependencies)) for c in sdk_definition.components))
        self.__dependencies = {}
        for comp, dep_ids in six.iteritems(deps):
            self.__dependencies[comp] = set((dep_id for dep_id in dep_ids if dep_id in deps))
        self.__consumers = dict(((id, set()) for id in self.__dependencies))
        for component_id, dep_ids in six.iteritems(self.__dependencies):
            for dep_id in dep_ids:
                self.__consumers[dep_id].add(component_id)

    def _ClosureFor(self, ids, adjacencies, component_filter):
        """Calculates a connected closure for the components with the given ids.

    Performs a breadth first search starting with the given component ids, and
    returns the set of components reachable via the given adjacency map.

    Args:
      ids: [str], The component ids to get the closure for.
      adjacencies: {str: set}, Map of component ids to the set of their
        adjacent component ids.
      component_filter: schemas.Component -> bool, A function applied to
        components that determines whether or not to include them in the
        closure.

    Returns:
      set of str, The set of component ids in the closure.
    """
        closure = set()
        to_process = collections.deque(ids)
        while to_process:
            current = to_process.popleft()
            if current not in self.components or current in closure:
                continue
            if not component_filter(self.components[current]):
                continue
            closure.add(current)
            to_process.extend(adjacencies[current])
        return closure

    def ComponentFromId(self, component_id):
        """Gets the schemas.Component from this snapshot with the given id.

    Args:
      component_id: str, The id component to get.

    Returns:
      The corresponding schemas.Component object.
    """
        return self.components.get(component_id)

    def ComponentsFromIds(self, component_ids):
        """Gets the schemas.Component objects for each of the given ids.

    Args:
      component_ids: iterable of str, The ids of the  components to get

    Returns:
      The corresponding schemas.Component objects.
    """
        return set((self.components.get(component_id) for component_id in component_ids))

    def AllComponentIdsMatching(self, platform_filter):
        """Gets all components in the snapshot that match the given platform.

    Args:
      platform_filter: platforms.Platform, A platform the components must match.

    Returns:
      set(str), The matching component ids.
    """
        return set((c_id for c_id, component in six.iteritems(self.components) if component.platform.Matches(platform_filter)))

    def DependencyClosureForComponents(self, component_ids, platform_filter=None):
        """Gets all the components that are depended on by any of the given ids.

    Args:
      component_ids: list of str, The ids of the components to get the
        dependencies of.
      platform_filter: platforms.Platform, A platform that components must
        match to be pulled into the dependency closure.

    Returns:
      set of str, All component ids that are in the dependency closure,
      including the given components.
    """
        component_filter = lambda c: c.platform.Matches(platform_filter)
        return self._ClosureFor(component_ids, self.__dependencies, component_filter)

    def ConsumerClosureForComponents(self, component_ids, platform_filter=None):
        """Gets all the components that depend on any of the given ids.

    Args:
      component_ids: list of str, The ids of the components to get the consumers
        of.
      platform_filter: platforms.Platform, A platform that components must
        match to be pulled into the consumer closure.

    Returns:
      set of str, All component ids that are in the consumer closure, including
      the given components.
    """
        component_filter = lambda c: c.platform.Matches(platform_filter)
        return self._ClosureFor(component_ids, self.__consumers, component_filter)

    def ConnectedComponents(self, component_ids, platform_filter=None):
        """Gets all the components that are connected to any of the given ids.

    Connected means in the connected graph of dependencies.  This is basically
    the union of the dependency and consumer closure for the given ids.

    Args:
      component_ids: list of str, The ids of the components to get the
        connected graph of.
      platform_filter: platforms.Platform, A platform that components must
        match to be pulled into the connected graph.

    Returns:
      set of str, All component ids that are connected to the given ids,
      including the given components.
    """
        adjacencies = {c_id: self.__dependencies[c_id] | self.__consumers[c_id] for c_id in self.components}
        component_filter = lambda c: c.platform.Matches(platform_filter)
        return self._ClosureFor(component_ids, adjacencies, component_filter)

    def StronglyConnectedComponents(self, component_id):
        """Gets the components strongly connected to the given component id.

    In other words, this functions returns the strongly connected "component" of
    the dependency graph for the given component id. In this context the
    strongly connected "component" is the set of ids whose components are both
    dependencies and consumers of the given component. In practice this can be
    used to determine the platform-specific subcomponent ID's for a given
    component ID, since they will always be mutually dependent.

    Args:
      component_id: str, The id of the component for which to get the strongly
        connected components.

    Returns:
      set of str, The ids of the components that are strongly connected to the
        component with the given id.
    """
        component_filter = lambda c: True
        dependency_closure = self._ClosureFor([component_id], self.__dependencies, component_filter)
        consumer_closure = self._ClosureFor([component_id], self.__consumers, component_filter)
        return dependency_closure & consumer_closure

    def GetEffectiveComponentSize(self, component_id, platform_filter=None):
        """Computes the effective size of the given component.

    If the component does not exist or does not exist on this platform, the size
    is 0.

    If it has data, just use the reported size of its data.

    If there is no data, report the total size of all its direct hidden
    dependencies (that are valid on this platform).  We don't include visible
    dependencies because they will show up in the list with their own size.

    This is a best effort estimation.  It is not easily possible to accurately
    report size in all situations because complex dependency graphs (between
    hidden and visible components, as well as circular dependencies) makes it
    infeasible to correctly show size when only displaying visible components.
    The goal is mainly to not show some components as having no size at all
    when they are wrappers around platform specific components.

    Args:
      component_id: str, The component to get the size for.
      platform_filter: platforms.Platform, A platform that components must
        match in order to be considered for any operations.

    Returns:
      int, The effective size of the component.
    """
        size = 0
        component = self.ComponentFromId(component_id)
        if component and component.platform.Matches(platform_filter):
            if component.data:
                return component.data.size
            deps = [self.ComponentFromId(d) for d in self.__dependencies[component_id]]
            deps = [d for d in deps if d.platform.Matches(platform_filter) and d.is_hidden and d.data]
            for d in deps:
                size += d.data.size or 0
        return size

    def CreateDiff(self, latest_snapshot, platform_filter=None):
        """Creates a ComponentSnapshotDiff based on this snapshot and the given one.

    Args:
      latest_snapshot: ComponentSnapshot, The latest state of the world that we
        want to compare to.
      platform_filter: platforms.Platform, A platform that components must
        match in order to be considered for any operations.

    Returns:
      A ComponentSnapshotDiff object.
    """
        return ComponentSnapshotDiff(self, latest_snapshot, platform_filter=platform_filter)

    def CreateComponentInfos(self, platform_filter=None):
        all_components = self.AllComponentIdsMatching(platform_filter)
        infos = [ComponentInfo(component_id, self, platform_filter=platform_filter) for component_id in all_components]
        return infos

    def WriteToFile(self, path, component_id=None):
        """Writes this snapshot back out to a JSON file.

    Args:
      path: str, The path of the file to write to.
      component_id: Limit snapshot to this component.
          If not specified all components are written out.

    Raises:
      ValueError: for non existent component_id.
    """
        sdk_def_dict = self.sdk_definition.ToDictionary()
        if component_id:
            component_dict = [c for c in sdk_def_dict['components'] if c['id'] == component_id]
            if not component_dict:
                raise ValueError('Component {} is not in this snapshot {}'.format(component_id, ','.join([c['id'] for c in sdk_def_dict['components']])))
            if 'data' in component_dict[0]:
                for f in list(component_dict[0]['data'].keys()):
                    if f not in ('contents_checksum', 'type', 'source'):
                        del component_dict[0]['data'][f]
                component_dict[0]['data']['source'] = ''
            sdk_def_dict['components'] = component_dict
            for key in list(sdk_def_dict.keys()):
                if key not in ('components', 'schema_version', 'revision', 'version'):
                    del sdk_def_dict[key]
        files.WriteFileContents(path, json.dumps(sdk_def_dict, indent=2, sort_keys=True, separators=(',', ': ')))

    def CheckMissingPlatformExecutable(self, component_ids, platform_filter=None):
        """Gets all the components that miss required platform-specific executables.

    Args:
      component_ids: list of str, The ids of the components to check for.
      platform_filter: platforms.Platform, A platform that components must
        match to be pulled into the dependency closure.

    Returns:
      set of str, All component ids that miss required platform-specific
        executables.
    """
        invalid_seeds = set()
        for c_id in component_ids:
            if c_id in self.components and (not self.components[c_id].platform.architectures) and (not self.components[c_id].platform.operating_systems) and self.components[c_id].platform_required:
                deps = self.DependencyClosureForComponents([c_id], platform_filter=platform_filter)
                qualified = [d for d in deps if str(d).startswith('{}-'.format(c_id))]
                if not qualified:
                    invalid_seeds.add(c_id)
        return invalid_seeds