from __future__ import absolute_import, division, print_function
import copy
class VarDict(object):
    """
    DEPRECATION WARNING

    This class is deprecated and will be removed in community.general 10.0.0
    Modules should use the VarDict from plugins/module_utils/vardict.py instead.
    """

    def __init__(self):
        self._data = dict()
        self._meta = dict()

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getattr__(self, item):
        try:
            return self._data[item]
        except KeyError:
            return getattr(self._data, item)

    def __setattr__(self, key, value):
        if key in ('_data', '_meta'):
            super(VarDict, self).__setattr__(key, value)
        else:
            self.set(key, value)

    def meta(self, name):
        return self._meta[name]

    def set_meta(self, name, **kwargs):
        self.meta(name).set(**kwargs)

    def set(self, name, value, **kwargs):
        if name in ('_data', '_meta'):
            raise ValueError('Names _data and _meta are reserved for use by ModuleHelper')
        self._data[name] = value
        if name in self._meta:
            meta = self.meta(name)
        else:
            meta = VarMeta(**kwargs)
        meta.set_value(value)
        self._meta[name] = meta

    def output(self):
        return dict(((k, v) for k, v in self._data.items() if self.meta(k).output))

    def diff(self):
        diff_results = [(k, self.meta(k).diff_result) for k in self._data]
        diff_results = [dr for dr in diff_results if dr[1] is not None]
        if diff_results:
            before = dict(((dr[0], dr[1]['before']) for dr in diff_results))
            after = dict(((dr[0], dr[1]['after']) for dr in diff_results))
            return {'before': before, 'after': after}
        return None

    def facts(self):
        facts_result = dict(((k, v) for k, v in self._data.items() if self._meta[k].fact))
        return facts_result if facts_result else None

    def change_vars(self):
        return [v for v in self._data if self.meta(v).change]

    def has_changed(self, v):
        return self._meta[v].has_changed