from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class FeatureIncrementalChartParser(IncrementalChartParser, FeatureChartParser):

    def __init__(self, grammar, strategy=BU_LC_INCREMENTAL_FEATURE_STRATEGY, trace_chart_width=20, chart_class=FeatureIncrementalChart, **parser_args):
        IncrementalChartParser.__init__(self, grammar, strategy=strategy, trace_chart_width=trace_chart_width, chart_class=chart_class, **parser_args)