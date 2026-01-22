from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class FeatureIncrementalTopDownChartParser(FeatureIncrementalChartParser):

    def __init__(self, grammar, **parser_args):
        FeatureIncrementalChartParser.__init__(self, grammar, TD_INCREMENTAL_FEATURE_STRATEGY, **parser_args)