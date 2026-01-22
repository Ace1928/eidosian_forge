from dulwich import errors as git_errors
from .. import errors as brz_errors
class NoPushSupport(brz_errors.BzrError):
    _fmt = 'Push is not yet supported from %(source)r to %(target)r using %(mapping)r for %(revision_id)r. Try dpush instead.'

    def __init__(self, source, target, mapping, revision_id=None):
        self.source = source
        self.target = target
        self.mapping = mapping
        self.revision_id = revision_id