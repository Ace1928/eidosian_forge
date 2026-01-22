from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class FeatureChartParser(ChartParser):

    def __init__(self, grammar, strategy=BU_LC_FEATURE_STRATEGY, trace_chart_width=20, chart_class=FeatureChart, **parser_args):
        ChartParser.__init__(self, grammar, strategy=strategy, trace_chart_width=trace_chart_width, chart_class=chart_class, **parser_args)