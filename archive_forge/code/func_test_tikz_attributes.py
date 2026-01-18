import pytest
import networkx as nx
def test_tikz_attributes():
    G = nx.path_graph(4, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in G}
    G.add_edge(0, 0)
    G.edges[0, 0]['label'] = 'Loop'
    G.edges[0, 0]['label_options'] = 'midway'
    G.nodes[0]['style'] = 'blue'
    G.nodes[1]['style'] = 'line width=3,draw'
    G.nodes[2]['style'] = 'circle,draw,blue!50'
    G.nodes[3]['label'] = 'Stop'
    G.edges[0, 1]['label'] = '1st Step'
    G.edges[0, 1]['label_options'] = 'near end'
    G.edges[2, 3]['label'] = '3rd Step'
    G.edges[2, 3]['label_options'] = 'near start'
    G.edges[2, 3]['style'] = 'bend left,green'
    G.edges[1, 2]['label'] = '2nd'
    G.edges[1, 2]['label_options'] = 'pos=0.5'
    G.edges[1, 2]['style'] = '>->,bend right,line width=3,green!90'
    output_tex = nx.to_latex(G, pos=pos, as_document=False, tikz_options='[scale=3]', node_options='style', edge_options='style', node_label='label', edge_label='label', edge_label_options='label_options')
    expected_tex = '\\begin{figure}\n  \\begin{tikzpicture}[scale=3]\n      \\draw\n        (0, 0) node[blue] (0){0}\n        (1, 1) node[line width=3,draw] (1){1}\n        (2, 2) node[circle,draw,blue!50] (2){2}\n        (3, 3) node (3){Stop};\n      \\begin{scope}[->]\n        \\draw (0) to node[near end] {1st Step} (1);\n        \\draw[loop,] (0) to node[midway] {Loop} (0);\n        \\draw[>->,bend right,line width=3,green!90] (1) to node[pos=0.5] {2nd} (2);\n        \\draw[bend left,green] (2) to node[near start] {3rd Step} (3);\n      \\end{scope}\n    \\end{tikzpicture}\n\\end{figure}'
    assert output_tex == expected_tex