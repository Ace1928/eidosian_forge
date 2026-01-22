from .. import config
class GitBranchConfig(config.BranchConfig):
    """BranchConfig that uses locations.conf in place of branch.conf"""

    def __init__(self, branch):
        super().__init__(branch)
        self.option_sources = (self.option_sources[0], self.option_sources[2])

    def __repr__(self):
        return '<{} of {!r}>'.format(self.__class__.__name__, self.branch)

    def set_user_option(self, name, value, store=config.STORE_BRANCH, warn_masked=False):
        """Force local to True"""
        config.BranchConfig.set_user_option(self, name, value, store=config.STORE_LOCATION, warn_masked=warn_masked)

    def _get_user_id(self):
        return self._get_best_value('_get_user_id')