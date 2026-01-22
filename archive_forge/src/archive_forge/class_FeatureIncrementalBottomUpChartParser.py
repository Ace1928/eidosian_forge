from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class FeatureIncrementalBottomUpChartParser(FeatureIncrementalChartParser):

    def __init__(self, grammar, **parser_args):
        FeatureIncrementalChartParser.__init__(self, grammar, BU_INCREMENTAL_FEATURE_STRATEGY, **parser_args)