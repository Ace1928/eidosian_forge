import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXProject(XCContainerPortal):
    """

  Attributes:
    path: "sample.xcodeproj".  TODO(mark) Document me!
    _other_pbxprojects: A dictionary, keyed by other PBXProject objects.  Each
                        value is a reference to the dict in the
                        projectReferences list associated with the keyed
                        PBXProject.
  """
    _schema = XCContainerPortal._schema.copy()
    _schema.update({'attributes': [0, dict, 0, 0], 'buildConfigurationList': [0, XCConfigurationList, 1, 1, XCConfigurationList()], 'compatibilityVersion': [0, str, 0, 1, 'Xcode 3.2'], 'hasScannedForEncodings': [0, int, 0, 1, 1], 'mainGroup': [0, PBXGroup, 1, 1, PBXGroup()], 'projectDirPath': [0, str, 0, 1, ''], 'projectReferences': [1, dict, 0, 0], 'projectRoot': [0, str, 0, 1, ''], 'targets': [1, XCTarget, 1, 1, []]})

    def __init__(self, properties=None, id=None, parent=None, path=None):
        self.path = path
        self._other_pbxprojects = {}
        return XCContainerPortal.__init__(self, properties, id, parent)

    def Name(self):
        name = self.path
        if name[-10:] == '.xcodeproj':
            name = name[:-10]
        return posixpath.basename(name)

    def Path(self):
        return self.path

    def Comment(self):
        return 'Project object'

    def Children(self):
        children = XCContainerPortal.Children(self)
        if 'projectReferences' in self._properties:
            for reference in self._properties['projectReferences']:
                children.append(reference['ProductGroup'])
        return children

    def PBXProjectAncestor(self):
        return self

    def _GroupByName(self, name):
        if 'mainGroup' not in self._properties:
            self.SetProperty('mainGroup', PBXGroup())
        main_group = self._properties['mainGroup']
        group = main_group.GetChildByName(name)
        if group is None:
            group = PBXGroup({'name': name})
            main_group.AppendChild(group)
        return group

    def SourceGroup(self):
        return self._GroupByName('Source')

    def ProductsGroup(self):
        return self._GroupByName('Products')

    def IntermediatesGroup(self):
        return self._GroupByName('Intermediates')

    def FrameworksGroup(self):
        return self._GroupByName('Frameworks')

    def ProjectsGroup(self):
        return self._GroupByName('Projects')

    def RootGroupForPath(self, path):
        """Returns a PBXGroup child of this object to which path should be added.

    This method is intended to choose between SourceGroup and
    IntermediatesGroup on the basis of whether path is present in a source
    directory or an intermediates directory.  For the purposes of this
    determination, any path located within a derived file directory such as
    PROJECT_DERIVED_FILE_DIR is treated as being in an intermediates
    directory.

    The returned value is a two-element tuple.  The first element is the
    PBXGroup, and the second element specifies whether that group should be
    organized hierarchically (True) or as a single flat list (False).
    """
        source_tree_groups = {'DERIVED_FILE_DIR': (self.IntermediatesGroup, True), 'INTERMEDIATE_DIR': (self.IntermediatesGroup, True), 'PROJECT_DERIVED_FILE_DIR': (self.IntermediatesGroup, True), 'SHARED_INTERMEDIATE_DIR': (self.IntermediatesGroup, True)}
        source_tree, path = SourceTreeAndPathFromPath(path)
        if source_tree is not None and source_tree in source_tree_groups:
            group_func, hierarchical = source_tree_groups[source_tree]
            group = group_func()
            return (group, hierarchical)
        return (self.SourceGroup(), True)

    def AddOrGetFileInRootGroup(self, path):
        """Returns a PBXFileReference corresponding to path in the correct group
    according to RootGroupForPath's heuristics.

    If an existing PBXFileReference for path exists, it will be returned.
    Otherwise, one will be created and returned.
    """
        group, hierarchical = self.RootGroupForPath(path)
        return group.AddOrGetFileByPath(path, hierarchical)

    def RootGroupsTakeOverOnlyChildren(self, recurse=False):
        """Calls TakeOverOnlyChild for all groups in the main group."""
        for group in self._properties['mainGroup']._properties['children']:
            if isinstance(group, PBXGroup):
                group.TakeOverOnlyChild(recurse)

    def SortGroups(self):
        self._properties['mainGroup']._properties['children'] = sorted(self._properties['mainGroup']._properties['children'], key=cmp_to_key(lambda x, y: x.CompareRootGroup(y)))
        for group in self._properties['mainGroup']._properties['children']:
            if not isinstance(group, PBXGroup):
                continue
            if group.Name() == 'Products':
                products = []
                for target in self._properties['targets']:
                    if not isinstance(target, PBXNativeTarget):
                        continue
                    product = target._properties['productReference']
                    assert product in group._properties['children']
                    products.append(product)
                assert len(products) == len(group._properties['children'])
                group._properties['children'] = products
            else:
                group.SortGroup()

    def AddOrGetProjectReference(self, other_pbxproject):
        """Add a reference to another project file (via PBXProject object) to this
    one.

    Returns [ProductGroup, ProjectRef].  ProductGroup is a PBXGroup object in
    this project file that contains a PBXReferenceProxy object for each
    product of each PBXNativeTarget in the other project file.  ProjectRef is
    a PBXFileReference to the other project file.

    If this project file already references the other project file, the
    existing ProductGroup and ProjectRef are returned.  The ProductGroup will
    still be updated if necessary.
    """
        if 'projectReferences' not in self._properties:
            self._properties['projectReferences'] = []
        product_group = None
        project_ref = None
        if other_pbxproject not in self._other_pbxprojects:
            product_group = PBXGroup({'name': 'Products'})
            product_group.parent = self
            product_group._hashables.extend(other_pbxproject.Hashables())
            this_path = posixpath.dirname(self.Path())
            projectDirPath = self.GetProperty('projectDirPath')
            if projectDirPath:
                if posixpath.isabs(projectDirPath[0]):
                    this_path = projectDirPath
                else:
                    this_path = posixpath.join(this_path, projectDirPath)
            other_path = gyp.common.RelativePath(other_pbxproject.Path(), this_path)
            project_ref = PBXFileReference({'lastKnownFileType': 'wrapper.pb-project', 'path': other_path, 'sourceTree': 'SOURCE_ROOT'})
            self.ProjectsGroup().AppendChild(project_ref)
            ref_dict = {'ProductGroup': product_group, 'ProjectRef': project_ref}
            self._other_pbxprojects[other_pbxproject] = ref_dict
            self.AppendProperty('projectReferences', ref_dict)
            self._properties['projectReferences'] = sorted(self._properties['projectReferences'], key=lambda x: x['ProjectRef'].Name().lower)
        else:
            project_ref_dict = self._other_pbxprojects[other_pbxproject]
            product_group = project_ref_dict['ProductGroup']
            project_ref = project_ref_dict['ProjectRef']
        self._SetUpProductReferences(other_pbxproject, product_group, project_ref)
        inherit_unique_symroot = self._AllSymrootsUnique(other_pbxproject, False)
        targets = other_pbxproject.GetProperty('targets')
        if all((self._AllSymrootsUnique(t, inherit_unique_symroot) for t in targets)):
            dir_path = project_ref._properties['path']
            product_group._hashables.extend(dir_path)
        return [product_group, project_ref]

    def _AllSymrootsUnique(self, target, inherit_unique_symroot):
        symroots = self._DefinedSymroots(target)
        for s in self._DefinedSymroots(target):
            if s is not None and (not self._IsUniqueSymrootForTarget(s)) or (s is None and (not inherit_unique_symroot)):
                return False
        return True if symroots else inherit_unique_symroot

    def _DefinedSymroots(self, target):
        config_list = target.GetProperty('buildConfigurationList')
        symroots = set()
        for config in config_list.GetProperty('buildConfigurations'):
            setting = config.GetProperty('buildSettings')
            if 'SYMROOT' in setting:
                symroots.add(setting['SYMROOT'])
            else:
                symroots.add(None)
        if len(symroots) == 1 and None in symroots:
            return set()
        return symroots

    def _IsUniqueSymrootForTarget(self, symroot):
        uniquifier = ['$SRCROOT', '$(SRCROOT)']
        if any((x in symroot for x in uniquifier)):
            return True
        return False

    def _SetUpProductReferences(self, other_pbxproject, product_group, project_ref):
        for target in other_pbxproject._properties['targets']:
            if not isinstance(target, PBXNativeTarget):
                continue
            other_fileref = target._properties['productReference']
            if product_group.GetChildByRemoteObject(other_fileref) is None:
                container_item = PBXContainerItemProxy({'containerPortal': project_ref, 'proxyType': 2, 'remoteGlobalIDString': other_fileref, 'remoteInfo': target.Name()})
                reference_proxy = PBXReferenceProxy({'fileType': other_fileref._properties['explicitFileType'], 'path': other_fileref._properties['path'], 'sourceTree': other_fileref._properties['sourceTree'], 'remoteRef': container_item})
                product_group.AppendChild(reference_proxy)

    def SortRemoteProductReferences(self):

        def CompareProducts(x, y, remote_products):
            x_remote = x._properties['remoteRef']._properties['remoteGlobalIDString']
            y_remote = y._properties['remoteRef']._properties['remoteGlobalIDString']
            x_index = remote_products.index(x_remote)
            y_index = remote_products.index(y_remote)
            return cmp(x_index, y_index)
        for other_pbxproject, ref_dict in self._other_pbxprojects.items():
            remote_products = []
            for target in other_pbxproject._properties['targets']:
                if not isinstance(target, PBXNativeTarget):
                    continue
                remote_products.append(target._properties['productReference'])
            product_group = ref_dict['ProductGroup']
            product_group._properties['children'] = sorted(product_group._properties['children'], key=cmp_to_key(lambda x, y, rp=remote_products: CompareProducts(x, y, rp)))