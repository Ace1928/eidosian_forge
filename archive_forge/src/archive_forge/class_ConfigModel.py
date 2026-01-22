from oslo_reports.models import with_default_views as mwdv
from oslo_reports.views.text import generic as generic_text_views
class ConfigModel(mwdv.ModelWithDefaultViews):
    """A Configuration Options Model

    This model holds data about a set of configuration options
    from :mod:`oslo_config`.  It supports both the default group
    of options and named option groups.

    :param conf_obj: a configuration object
    :type conf_obj: :class:`oslo_config.cfg.ConfigOpts`
    """

    def __init__(self, conf_obj):
        kv_view = generic_text_views.KeyValueView(dict_sep=': ', before_dict='')
        super(ConfigModel, self).__init__(text_view=kv_view)

        def opt_title(optname, co):
            return co._opts[optname]['opt'].name

        def opt_value(opt_obj, value):
            if opt_obj['opt'].secret:
                return '***'
            else:
                return value
        self['default'] = dict(((opt_title(optname, conf_obj), opt_value(conf_obj._opts[optname], conf_obj[optname])) for optname in conf_obj._opts))
        groups = {}
        for groupname in conf_obj._groups:
            group_obj = conf_obj._groups[groupname]
            curr_group_opts = dict(((opt_title(optname, group_obj), opt_value(group_obj._opts[optname], conf_obj[groupname][optname])) for optname in group_obj._opts))
            groups[group_obj.name] = curr_group_opts
        self.update(groups)