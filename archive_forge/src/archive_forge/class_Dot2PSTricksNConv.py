import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
class Dot2PSTricksNConv(Dot2PSTricksConv):
    """A backend that utilizes the node and edge mechanism of PSTricks-Node"""

    def __init__(self, options=None):
        options = options or {}
        options['switchdraworder'] = True
        options['flattengraph'] = True
        options['rawdim'] = True
        self.template = PSTRICKSN_TEMPLATE
        Dot2PSTricksConv.__init__(self, options)

    def output_node_comment(self, node):
        return ''

    def do_nodes(self):
        s = ''
        for node in self.nodes:
            self.currentnode = node
            psshadeoption = getattr(node, 'psshadeoption', '')
            psshape = getattr(node, 'psshape', '')
            if len(psshape) == 0:
                shape = getattr(node, 'shape', 'ellipse')
                psshape = 'psframebox'
                if shape == 'circle':
                    psshape = 'pscirclebox'
                if shape == 'ellipse':
                    psshape = 'psovalbox'
                if shape == 'triangle':
                    psshape = 'pstribox'
            width = getattr(node, 'width', '1')
            height = getattr(node, 'height', '1')
            psbox = getattr(node, 'psbox', 'false')
            color = getattr(node, 'color', '')
            fillcolor = getattr(node, 'fillcolor', '')
            if len(color) > 0:
                psshadeoption = 'linecolor=' + color + ',' + psshadeoption
            if len(fillcolor) > 0:
                psshadeoption = 'fillcolor=' + fillcolor + ',' + psshadeoption
            style = getattr(node, 'style', '')
            if len(style) > 0:
                if style == 'dotted':
                    psshadeoption = 'linestyle=dotted,' + psshadeoption
                if style == 'dashed':
                    psshadeoption = 'linestyle=dashed,' + psshadeoption
                if style == 'solid':
                    psshadeoption = 'linestyle=solid,' + psshadeoption
                if style == 'bold':
                    psshadeoption = 'linewidth=2pt,' + psshadeoption
            pos = getattr(node, 'pos')
            if not pos:
                continue
            x, y = pos.split(',')
            label = self.get_label(node)
            pos = '%sbp,%sbp' % (smart_float(x), smart_float(y))
            sn = ''
            sn += self.output_node_comment(node)
            sn += self.start_node(node)
            if psbox == 'false':
                sn += '\\rput(%s){\\rnode{%s}{\\%s[%s]{%s}}}\n' % (pos, tikzify(node.name), psshape, psshadeoption, label)
            else:
                sn += '\\rput(%s){\\rnode{%s}{\\%s[%s]{\\parbox[c][%sin][c]{%sin}{\\centering %s}}}}\n' % (pos, tikzify(node.name), psshape, psshadeoption, height, width, label)
            sn += self.end_node(node)
            s += sn
        self.body += s

    def do_edges(self):
        s = ''
        for edge in self.edges:
            s += self.draw_edge(edge)
        self.body += s

    def draw_edge(self, edge):
        s = ''
        edges = self.get_edge_points(edge)
        for arrowstyle, points in edges:
            psarrow = getattr(edge, 'psarrow', '')
            if len(psarrow) == 0:
                stylestr = '-'
            else:
                stylestr = psarrow
            psedge = getattr(edge, 'psedge', 'ncline')
            psedgeoption = getattr(edge, 'psedgeoption', '')
            color = getattr(edge, 'color', '')
            fillcolor = getattr(edge, 'fillcolor', '')
            if len(color) > 0:
                psedgeoption = 'linecolor=' + color + ',' + psedgeoption
            if len(fillcolor) > 0:
                psedgeoption = 'fillcolor=' + fillcolor + ',' + psedgeoption
            style = getattr(edge, 'style', '')
            if len(style) > 0:
                if style == 'dotted':
                    psedgeoption = 'linestyle=dotted,' + psedgeoption
                if style == 'dashed':
                    psedgeoption = 'linestyle=dashed,' + psedgeoption
                if style == 'solid':
                    psedgeoption = 'linestyle=solid,' + psedgeoption
                if style == 'bold':
                    psedgeoption = 'linewidth=2pt,' + psedgeoption
            pslabel = getattr(edge, 'pslabel', 'ncput')
            pslabeloption = getattr(edge, 'pslabeloption', '')
            label = getattr(edge, 'label', '')
            headlabel = getattr(edge, 'headlabel', '')
            taillabel = getattr(edge, 'taillabel', '')
            src = tikzify(edge.get_source())
            dst = tikzify(edge.get_destination())
            s = '\\%s[%s]{%s}{%s}{%s}\n' % (psedge, psedgeoption, stylestr, src, dst)
            if len(label) != 0:
                s += '\\%s[%s]{%s}\n' % (pslabel, pslabeloption, label)
            if len(headlabel) != 0:
                pslabelhead = 'npos=0.8,' + pslabeloption
                s += '\\%s[%s]{%s}\n' % (pslabel, pslabelhead, headlabel)
            if len(taillabel) != 0:
                pslabeltail = 'npos=0.2,' + pslabeloption
                s += '\\%s[%s]{%s}\n' % (pslabel, pslabeltail, taillabel)
        return s

    def start_node(self, node):
        return ''

    def end_node(self, node):
        return ''