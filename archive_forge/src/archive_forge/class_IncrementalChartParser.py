from time import perf_counter
from nltk.parse.chart import (
from nltk.parse.featurechart import (
class IncrementalChartParser(ChartParser):
    """
    An *incremental* chart parser implementing Jay Earley's
    parsing algorithm:

    | For each index end in [0, 1, ..., N]:
    |   For each edge such that edge.end = end:
    |     If edge is incomplete and edge.next is not a part of speech:
    |       Apply PredictorRule to edge
    |     If edge is incomplete and edge.next is a part of speech:
    |       Apply ScannerRule to edge
    |     If edge is complete:
    |       Apply CompleterRule to edge
    | Return any complete parses in the chart
    """

    def __init__(self, grammar, strategy=BU_LC_INCREMENTAL_STRATEGY, trace=0, trace_chart_width=50, chart_class=IncrementalChart):
        """
        Create a new Earley chart parser, that uses ``grammar`` to
        parse texts.

        :type grammar: CFG
        :param grammar: The grammar used to parse texts.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            and higher numbers will produce more verbose tracing
            output.
        :type trace_chart_width: int
        :param trace_chart_width: The default total width reserved for
            the chart in trace output.  The remainder of each line will
            be used to display edges.
        :param chart_class: The class that should be used to create
            the charts used by this parser.
        """
        self._grammar = grammar
        self._trace = trace
        self._trace_chart_width = trace_chart_width
        self._chart_class = chart_class
        self._axioms = []
        self._inference_rules = []
        for rule in strategy:
            if rule.NUM_EDGES == 0:
                self._axioms.append(rule)
            elif rule.NUM_EDGES == 1:
                self._inference_rules.append(rule)
            else:
                raise ValueError('Incremental inference rules must have NUM_EDGES == 0 or 1')

    def chart_parse(self, tokens, trace=None):
        if trace is None:
            trace = self._trace
        trace_new_edges = self._trace_new_edges
        tokens = list(tokens)
        self._grammar.check_coverage(tokens)
        chart = self._chart_class(tokens)
        grammar = self._grammar
        trace_edge_width = self._trace_chart_width // (chart.num_leaves() + 1)
        if trace:
            print(chart.pretty_format_leaves(trace_edge_width))
        for axiom in self._axioms:
            new_edges = list(axiom.apply(chart, grammar))
            trace_new_edges(chart, axiom, new_edges, trace, trace_edge_width)
        inference_rules = self._inference_rules
        for end in range(chart.num_leaves() + 1):
            if trace > 1:
                print('\n* Processing queue:', end, '\n')
            agenda = list(chart.select(end=end))
            while agenda:
                edge = agenda.pop()
                for rule in inference_rules:
                    new_edges = list(rule.apply(chart, grammar, edge))
                    trace_new_edges(chart, rule, new_edges, trace, trace_edge_width)
                    for new_edge in new_edges:
                        if new_edge.end() == end:
                            agenda.append(new_edge)
        return chart