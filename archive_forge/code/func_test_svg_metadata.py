import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree
import xml.parsers.expat
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib import font_manager as fm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
def test_svg_metadata():
    single_value = ['Coverage', 'Identifier', 'Language', 'Relation', 'Source', 'Title', 'Type']
    multi_value = ['Contributor', 'Creator', 'Keywords', 'Publisher', 'Rights']
    metadata = {'Date': [datetime.date(1968, 8, 1), datetime.datetime(1968, 8, 2, 1, 2, 3)], 'Description': 'description\ntext', **{k: f'{k} foo' for k in single_value}, **{k: [f'{k} bar', f'{k} baz'] for k in multi_value}}
    fig = plt.figure()
    with BytesIO() as fd:
        fig.savefig(fd, format='svg', metadata=metadata)
        buf = fd.getvalue().decode()
    SVGNS = '{http://www.w3.org/2000/svg}'
    RDFNS = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
    CCNS = '{http://creativecommons.org/ns#}'
    DCNS = '{http://purl.org/dc/elements/1.1/}'
    root = xml.etree.ElementTree.fromstring(buf)
    rdf, = root.findall(f'./{SVGNS}metadata/{RDFNS}RDF')
    titles = [node.text for node in root.findall(f'./{SVGNS}title')]
    assert titles == [metadata['Title']]
    types = [node.attrib[f'{RDFNS}resource'] for node in rdf.findall(f'./{CCNS}Work/{DCNS}type')]
    assert types == [metadata['Type']]
    for k in ['Description', *single_value]:
        if k == 'Type':
            continue
        values = [node.text for node in rdf.findall(f'./{CCNS}Work/{DCNS}{k.lower()}')]
        assert values == [metadata[k]]
    for k in multi_value:
        if k == 'Keywords':
            continue
        values = [node.text for node in rdf.findall(f'./{CCNS}Work/{DCNS}{k.lower()}/{CCNS}Agent/{DCNS}title')]
        assert values == metadata[k]
    dates = [node.text for node in rdf.findall(f'./{CCNS}Work/{DCNS}date')]
    assert dates == ['1968-08-01/1968-08-02T01:02:03']
    values = [node.text for node in rdf.findall(f'./{CCNS}Work/{DCNS}subject/{RDFNS}Bag/{RDFNS}li')]
    assert values == metadata['Keywords']