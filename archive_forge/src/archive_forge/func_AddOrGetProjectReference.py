import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
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