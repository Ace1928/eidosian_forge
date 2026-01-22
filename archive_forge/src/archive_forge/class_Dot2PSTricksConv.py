import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
class Dot2PSTricksConv(DotConvBase):
    """PSTricks converter backend"""

    def __init__(self, options=None):
        DotConvBase.__init__(self, options)
        if not self.template:
            self.template = PSTRICKS_TEMPLATE
        self.styles = dict(dotted='linestyle=dotted', dashed='linestyle=dashed', bold='linewidth=2pt', solid='', filled='')

    def do_graphtmp(self):
        self.pencolor = ''
        self.fillcolor = ''
        self.color = ''
        self.body += '{\n'
        DotConvBase.do_graph(self)
        self.body += '}\n'

    def draw_ellipse(self, drawop, style=None):
        op, x, y, w, h = drawop
        s = ''
        if op == 'E':
            if style:
                style = style.replace('filled', '')
            stylestr = 'fillstyle=solid'
        else:
            stylestr = ''
        if style:
            if stylestr:
                stylestr += ',' + style
            else:
                stylestr = style
        s += '  \\psellipse[%s](%sbp,%sbp)(%sbp,%sbp)\n' % (stylestr, smart_float(x), smart_float(y), smart_float(w), smart_float(h))
        return s

    def draw_polygon(self, drawop, style=None):
        op, points = drawop
        pp = ['(%sbp,%sbp)' % (smart_float(p[0]), smart_float(p[1])) for p in points]
        stylestr = ''
        if op == 'P':
            if style:
                style = style.replace('filled', '')
            stylestr = 'fillstyle=solid'
        if style:
            if stylestr:
                stylestr += ',' + style
            else:
                stylestr = style
        s = '  \\pspolygon[%s]%s\n' % (stylestr, ''.join(pp))
        return s

    def draw_polyline(self, drawop, style=None):
        op, points = drawop
        pp = ['(%sbp,%sbp)' % (smart_float(p[0]), smart_float(p[1])) for p in points]
        s = '  \\psline%s\n' % ''.join(pp)
        return s

    def draw_bezier(self, drawop, style=None):
        op, points = drawop
        pp = []
        for point in points:
            pp.append('(%sbp,%sbp)' % (smart_float(point[0]), smart_float(point[1])))
        arrowstyle = ''
        return '  \\psbezier{%s}%s\n' % (arrowstyle, ''.join(pp))

    def draw_text(self, drawop, style=None):
        if len(drawop) == 7:
            c, x, y, align, w, text, valign = drawop
        else:
            c, x, y, align, w, text = drawop
            valign = ''
        if align == '-1':
            alignstr = 'l'
        elif align == '1':
            alignstr = 'r'
        else:
            alignstr = ''
        if alignstr or valign:
            alignstr = '[' + alignstr + valign + ']'
        s = '  \\rput%s(%sbp,%sbp){%s}\n' % (alignstr, smart_float(x), smart_float(y), text)
        return s

    def set_color(self, drawop):
        c, color = drawop
        color = self.convert_color(color)
        s = ''
        if c == 'c':
            if self.pencolor != color:
                self.pencolor = color
                s = '  \\psset{linecolor=%s}\n' % color
            else:
                return ''
        elif c == 'C':
            if self.fillcolor != color:
                self.fillcolor = color
                s = '  \\psset{fillcolor=%s}\n' % color
            else:
                return ''
        elif c == 'cC':
            if self.color != color:
                self.color = color
                self.pencolor = self.fillcolor = color
                s = '  \\psset{linecolor=%s}\n' % color
        else:
            log.warning('Unhandled color: %s', drawop)
        return s

    def set_style(self, drawop):
        c, style = drawop
        psstyle = self.styles.get(style, '')
        if psstyle:
            return '  \\psset{%s}\n' % psstyle
        else:
            return ''

    def filter_styles(self, style):
        filtered_styles = []
        for item in style.split(','):
            keyval = item.strip()
            if keyval.find('setlinewidth') < 0:
                filtered_styles.append(keyval)
        return ', '.join(filtered_styles)

    def start_node(self, node):
        self.pencolor = ''
        self.fillcolor = ''
        self.color = ''
        return '{%\n'

    def end_node(self, node):
        return '}%\n'

    def start_edge(self):
        self.pencolor = ''
        self.fillcolor = ''
        return '{%\n'

    def end_edge(self):
        return '}%\n'

    def start_graph(self, graph):
        self.pencolor = ''
        self.fillcolor = ''
        self.color = ''
        return '{\n'

    def end_graph(self, node):
        return '}\n'

    def draw_edge(self, edge):
        s = ''
        if edge.attr.get('style', '') in ['invis', 'invisible']:
            return ''
        edges = self.get_edge_points(edge)
        for arrowstyle, points in edges:
            if arrowstyle == '--':
                arrowstyle = ''
            color = getattr(edge, 'color', '')
            if self.color != color:
                if color:
                    s += self.set_color(('c', color))
                else:
                    s += self.set_color(('c', 'black'))
            pp = []
            for point in points:
                p = point.split(',')
                pp.append('(%sbp,%sbp)' % (smart_float(p[0]), smart_float(p[1])))
            edgestyle = edge.attr.get('style', '')
            styles = []
            if arrowstyle:
                styles.append('arrows=%s' % arrowstyle)
            if edgestyle:
                edgestyles = [self.styles.get(key.strip(), key.strip()) for key in edgestyle.split(',') if key]
                styles.extend(edgestyles)
            if styles:
                stylestr = ','.join(styles)
            else:
                stylestr = ''
            if not self.options.get('straightedges'):
                s += '  \\psbezier[%s]%s\n' % (stylestr, ''.join(pp))
            else:
                s += '  \\psline[%s]%s%s\n' % (stylestr, pp[0], pp[-1])
        return s

    def init_template_vars(self):
        DotConvBase.init_template_vars(self)
        graphstyle = self.templatevars.get('<<graphstyle>>', '')
        if graphstyle:
            graphstyle = graphstyle.strip()
            if not graphstyle.startswith(','):
                graphstyle = ',' + graphstyle
                self.templatevars['<<graphstyle>>'] = graphstyle