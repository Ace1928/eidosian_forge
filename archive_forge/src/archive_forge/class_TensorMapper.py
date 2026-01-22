import json
import os
import re
import sys
import numpy as np
class TensorMapper:
    """Maps a list of tensor indices to a tooltip hoverable indicator of more."""

    def __init__(self, subgraph_data):
        self.data = subgraph_data

    def __call__(self, x):
        html = ''
        if x is None:
            return html
        html += "<span class='tooltip'><span class='tooltipcontent'>"
        for i in x:
            tensor = self.data['tensors'][i]
            html += str(i) + ' '
            html += NameListToString(tensor['name']) + ' '
            html += TensorTypeToName(tensor['type']) + ' '
            html += repr(tensor['shape']) if 'shape' in tensor else '[]'
            html += (repr(tensor['shape_signature']) if 'shape_signature' in tensor else '[]') + '<br>'
        html += '</span>'
        html += repr(x)
        html += '</span>'
        return html