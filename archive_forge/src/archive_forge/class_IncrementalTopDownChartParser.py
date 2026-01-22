from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class IncrementalTopDownChartParser(IncrementalChartParser):

    def __init__(self, grammar, **parser_args):
        IncrementalChartParser.__init__(self, grammar, TD_INCREMENTAL_STRATEGY, **parser_args)