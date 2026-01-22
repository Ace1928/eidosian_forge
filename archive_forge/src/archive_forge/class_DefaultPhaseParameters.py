from sys import version_info as _swig_python_version_info
import weakref
class DefaultPhaseParameters(object):
    """
    This struct holds all parameters for the default search.
    DefaultPhaseParameters is only used by Solver::MakeDefaultPhase methods.
    Note this is for advanced users only.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    CHOOSE_MAX_SUM_IMPACT = _pywrapcp.DefaultPhaseParameters_CHOOSE_MAX_SUM_IMPACT
    CHOOSE_MAX_AVERAGE_IMPACT = _pywrapcp.DefaultPhaseParameters_CHOOSE_MAX_AVERAGE_IMPACT
    CHOOSE_MAX_VALUE_IMPACT = _pywrapcp.DefaultPhaseParameters_CHOOSE_MAX_VALUE_IMPACT
    SELECT_MIN_IMPACT = _pywrapcp.DefaultPhaseParameters_SELECT_MIN_IMPACT
    SELECT_MAX_IMPACT = _pywrapcp.DefaultPhaseParameters_SELECT_MAX_IMPACT
    NONE = _pywrapcp.DefaultPhaseParameters_NONE
    NORMAL = _pywrapcp.DefaultPhaseParameters_NORMAL
    VERBOSE = _pywrapcp.DefaultPhaseParameters_VERBOSE
    var_selection_schema = property(_pywrapcp.DefaultPhaseParameters_var_selection_schema_get, _pywrapcp.DefaultPhaseParameters_var_selection_schema_set, doc='\n    This parameter describes how the next variable to instantiate\n    will be chosen.\n    ')
    value_selection_schema = property(_pywrapcp.DefaultPhaseParameters_value_selection_schema_get, _pywrapcp.DefaultPhaseParameters_value_selection_schema_set, doc=' This parameter describes which value to select for a given var.')
    initialization_splits = property(_pywrapcp.DefaultPhaseParameters_initialization_splits_get, _pywrapcp.DefaultPhaseParameters_initialization_splits_set, doc='\n    Maximum number of intervals that the initialization of impacts will scan\n    per variable.\n    ')
    run_all_heuristics = property(_pywrapcp.DefaultPhaseParameters_run_all_heuristics_get, _pywrapcp.DefaultPhaseParameters_run_all_heuristics_set, doc='\n    The default phase will run heuristics periodically. This parameter\n    indicates if we should run all heuristics, or a randomly selected\n    one.\n    ')
    heuristic_period = property(_pywrapcp.DefaultPhaseParameters_heuristic_period_get, _pywrapcp.DefaultPhaseParameters_heuristic_period_set, doc='\n    The distance in nodes between each run of the heuristics. A\n    negative or null value will mean that we will not run heuristics\n    at all.\n    ')
    heuristic_num_failures_limit = property(_pywrapcp.DefaultPhaseParameters_heuristic_num_failures_limit_get, _pywrapcp.DefaultPhaseParameters_heuristic_num_failures_limit_set, doc=' The failure limit for each heuristic that we run.')
    persistent_impact = property(_pywrapcp.DefaultPhaseParameters_persistent_impact_get, _pywrapcp.DefaultPhaseParameters_persistent_impact_set, doc='\n    Whether to keep the impact from the first search for other searches,\n    or to recompute the impact for each new search.\n    ')
    random_seed = property(_pywrapcp.DefaultPhaseParameters_random_seed_get, _pywrapcp.DefaultPhaseParameters_random_seed_set, doc=' Seed used to initialize the random part in some heuristics.')
    display_level = property(_pywrapcp.DefaultPhaseParameters_display_level_get, _pywrapcp.DefaultPhaseParameters_display_level_set, doc='\n    This represents the amount of information displayed by the default search.\n    NONE means no display, VERBOSE means extra information.\n    ')
    decision_builder = property(_pywrapcp.DefaultPhaseParameters_decision_builder_get, _pywrapcp.DefaultPhaseParameters_decision_builder_set, doc=' When defined, this overrides the default impact based decision builder.')

    def __init__(self):
        _pywrapcp.DefaultPhaseParameters_swiginit(self, _pywrapcp.new_DefaultPhaseParameters())
    __swig_destroy__ = _pywrapcp.delete_DefaultPhaseParameters