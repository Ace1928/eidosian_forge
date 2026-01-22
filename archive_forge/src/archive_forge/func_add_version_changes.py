from collections import namedtuple
def add_version_changes(library, version, structure, fields, removals, repositions=None):
    if version not in _version_changes[library]:
        _version_changes[library][version] = {}
    if structure in _version_changes[library][version]:
        raise Exception('Structure: {} from: {} has already been added for version {}.'.format(structure, library, version))
    _version_changes[library][version][structure] = CustomField(fields, removals, repositions)