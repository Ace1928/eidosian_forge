from __future__ import absolute_import, division, print_function
import copy
class VarsMixin(object):
    """
    DEPRECATION WARNING

    This class is deprecated and will be removed in community.general 10.0.0
    Modules should use the VarDict from plugins/module_utils/vardict.py instead.
    """

    def __init__(self, module=None):
        self.vars = VarDict()
        super(VarsMixin, self).__init__(module)

    def update_vars(self, meta=None, **kwargs):
        if meta is None:
            meta = {}
        for k, v in kwargs.items():
            self.vars.set(k, v, **meta)