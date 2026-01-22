from . import errors, registry, urlutils
class UnknownBugTrackerAbbreviation(errors.BzrError):
    _fmt = 'Cannot find registered bug tracker called %(abbreviation)s on %(branch)s'

    def __init__(self, abbreviation, branch):
        self.abbreviation = abbreviation
        self.branch = branch