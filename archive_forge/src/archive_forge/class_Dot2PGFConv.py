import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
class Dot2PGFConv(DotConvBase):
    """PGF/TikZ converter backend"""
    arrows_map_210 = {'dot': '*', 'odot': 'o', 'empty': 'open triangle 45', 'invempty': 'open triangle 45 reversed', 'diamond': 'diamond', 'odiamond': 'open diamond', 'ediamond': 'open diamond', 'box': 'square', 'obox': 'open square', 'vee': "stealth'", 'open': "stealth'", 'tee': '|', 'crow': 'stealth reversed'}

    def __init__(self, options=None):
        DotConvBase.__init__(self, options)
        if not self.template:
            if options.get('pgf118'):
                self.template = PGF118_TEMPLATE
            elif options.get('pgf210'):
                self.template = PGF210_TEMPLATE
            else:
                self.template = PGF_TEMPLATE
        self.styles = dict(dashed='dashed', dotted='dotted', bold='very thick', filled='fill', invis='', rounded='rounded corners')
        self.dashstyles = dict(dashed='\\pgfsetdash{{3pt}{3pt}}{0pt}', dotted='\\pgfsetdash{{\\pgflinewidth}{2pt}}{0pt}', bold='\\pgfsetlinewidth{1.2pt}')

    def start_node(self, node):
        self.pencolor = ''
        self.fillcolor = ''
        self.color = ''
        return '\\begin{scope}\n'

    def end_node(self, node):
        return '\\end{scope}\n'

    def start_edge(self):
        return '\\begin{scope}\n'

    def end_edge(self):
        return '\\end{scope}\n'

    def start_graph(self, graph):
        self.pencolor = ''
        self.fillcolor = ''
        self.color = ''
        return '\\begin{scope}\n'

    def end_graph(self, graph):
        return '\\end{scope}\n'

    def set_color(self, drawop):
        c, color = drawop
        res = self.convert_color(color, True)
        opacity = None
        if len(res) == 2:
            ccolor, opacity = res
        else:
            ccolor = res
        s = ''
        if c == 'cC':
            if self.color != color:
                self.color = color
                self.pencolor = color
                self.fillcolor = color
                if ccolor.startswith('{'):
                    s += '  \\definecolor{newcol}%s;\n' % ccolor
                    ccolor = 'newcol'
                s += '  \\pgfsetcolor{%s}\n' % ccolor
        elif c == 'c':
            if self.pencolor != color:
                self.pencolor = color
                self.color = ''
                if ccolor.startswith('{'):
                    s += '  \\definecolor{strokecol}%s;\n' % ccolor
                    ccolor = 'strokecol'
                s += '  \\pgfsetstrokecolor{%s}\n' % ccolor
            else:
                return ''
        elif c == 'C':
            if self.fillcolor != color:
                self.fillcolor = color
                self.color = ''
                if ccolor.startswith('{'):
                    s += '  \\definecolor{fillcol}%s;\n' % ccolor
                    ccolor = 'fillcol'
                s += '  \\pgfsetfillcolor{%s}\n' % ccolor
                if not opacity is None:
                    self.opacity = opacity
                else:
                    self.opacity = None
            else:
                return ''
        return s

    def set_style(self, drawop):
        c, style = drawop
        pgfstyle = self.dashstyles.get(style, '')
        if pgfstyle:
            return '  %s\n' % pgfstyle
        else:
            return ''

    def filter_styles(self, style):
        filtered_styles = []
        for item in style.split(','):
            keyval = item.strip()
            if keyval.find('setlinewidth') < 0 and (not keyval == 'filled'):
                filtered_styles.append(keyval)
        return ', '.join(filtered_styles)

    def draw_ellipse(self, drawop, style=None):
        op, x, y, w, h = drawop
        s = ''
        if op == 'E':
            if self.opacity is not None:
                cmd = 'filldraw [opacity=%s]' % self.opacity
            else:
                cmd = 'filldraw'
        else:
            cmd = 'draw'
        if style:
            stylestr = ' [%s]' % style
        else:
            stylestr = ''
        s += '  \\%s%s (%sbp,%sbp) ellipse (%sbp and %sbp);\n' % (cmd, stylestr, smart_float(x), smart_float(y), smart_float(w), smart_float(h))
        return s

    def draw_polygon(self, drawop, style=None):
        op, points = drawop
        pp = ['(%sbp,%sbp)' % (smart_float(p[0]), smart_float(p[1])) for p in points]
        cmd = 'draw'
        if op == 'P':
            cmd = 'filldraw'
        if style:
            stylestr = ' [%s]' % style
        else:
            stylestr = ''
        s = '  \\%s%s %s -- cycle;\n' % (cmd, stylestr, ' -- '.join(pp))
        return s

    def draw_polyline(self, drawop, style=None):
        op, points = drawop
        pp = ['(%sbp,%sbp)' % (smart_float(p[0]), smart_float(p[1])) for p in points]
        stylestr = ''
        return '  \\draw%s %s;\n' % (stylestr, ' -- '.join(pp))

    def draw_text(self, drawop, style=None):
        if len(drawop) == 7:
            c, x, y, align, w, text, valign = drawop
        else:
            c, x, y, align, w, text = drawop
        styles = []
        if align == '-1':
            alignstr = 'right'
        elif align == '1':
            alignstr = 'left'
        else:
            alignstr = ''
        styles.append(alignstr)
        styles.append(style)
        lblstyle = ','.join([i for i in styles if i])
        if lblstyle:
            lblstyle = '[' + lblstyle + ']'
        s = '  \\draw (%sbp,%sbp) node%s {%s};\n' % (smart_float(x), smart_float(y), lblstyle, text)
        return s

    def draw_bezier(self, drawop, style=None):
        s = ''
        c, points = drawop
        pp = []
        for point in points:
            pp.append('(%sbp,%sbp)' % (smart_float(point[0]), smart_float(point[1])))
        pstrs = ['%s .. controls %s and %s ' % p for p in nsplit(pp, 3)]
        stylestr = ''
        s += '  \\draw%s %s .. %s;\n' % (stylestr, ' .. '.join(pstrs), pp[-1])
        return s

    def do_edges(self):
        s = ''
        s += self.set_color(('cC', 'black'))
        for edge in self.edges:
            general_draw_string = getattr(edge, '_draw_', '')
            label_string = getattr(edge, '_ldraw_', '')
            head_arrow_string = getattr(edge, '_hdraw_', '')
            tail_arrow_string = getattr(edge, '_tdraw_', '')
            tail_label_string = getattr(edge, '_tldraw_', '')
            head_label_string = getattr(edge, '_hldraw_', '')
            drawstring = general_draw_string + ' ' + head_arrow_string + ' ' + tail_arrow_string + ' ' + label_string
            draw_operations, stat = parse_drawstring(drawstring)
            if not drawstring.strip():
                continue
            s += self.output_edge_comment(edge)
            if self.options.get('duplicate'):
                s += self.start_edge()
                s += self.do_draw_op(draw_operations, edge, stat)
                s += self.do_drawstring(tail_label_string, edge, 'tailtexlbl')
                s += self.do_drawstring(head_label_string, edge, 'headtexlbl')
                s += self.end_edge()
            else:
                topath = getattr(edge, 'topath', None)
                s += self.draw_edge(edge)
                if not self.options.get('tikzedgelabels') and (not topath):
                    s += self.do_drawstring(label_string, edge)
                    s += self.do_drawstring(tail_label_string, edge, 'tailtexlbl')
                    s += self.do_drawstring(head_label_string, edge, 'headtexlbl')
                else:
                    s += self.do_drawstring(tail_label_string, edge, 'tailtexlbl')
                    s += self.do_drawstring(head_label_string, edge, 'headtexlbl')
        self.body += s

    def draw_edge(self, edge):
        s = ''
        if edge.attr.get('style', '') in ['invis', 'invisible']:
            return ''
        edges = self.get_edge_points(edge)
        for arrowstyle, points in edges:
            color = getattr(edge, 'color', '')
            if self.color != color:
                if color:
                    s += self.set_color(('cC', color))
                else:
                    s += self.set_color(('cC', 'black'))
            pp = []
            for point in points:
                p = point.split(',')
                pp.append('(%sbp,%sbp)' % (smart_float(p[0]), smart_float(p[1])))
            edgestyle = edge.attr.get('style', '')
            styles = []
            if arrowstyle != '--':
                styles = [arrowstyle]
            if edgestyle:
                edgestyles = [self.styles.get(key.strip(), key.strip()) for key in edgestyle.split(',') if key]
                styles.extend(edgestyles)
            stylestr = ','.join(styles)
            topath = getattr(edge, 'topath', None)
            pstrs = ['%s .. controls %s and %s ' % x for x in nsplit(pp, 3)]
            extra = ''
            if self.options.get('tikzedgelabels') or topath:
                edgelabel = self.get_label(edge)
                lblstyle = getattr(edge, 'lblstyle', '')
                if lblstyle:
                    lblstyle = '[' + lblstyle + ']'
                else:
                    lblstyle = ''
                if edgelabel:
                    extra = ' node%s {%s}' % (lblstyle, edgelabel)
            src = pp[0]
            dst = pp[-1]
            if topath:
                s += '  \\draw [%s] %s to[%s]%s %s;\n' % (stylestr, src, topath, extra, dst)
            elif not self.options.get('straightedges'):
                s += '  \\draw [%s] %s ..%s %s;\n' % (stylestr, ' .. '.join(pstrs), extra, pp[-1])
            else:
                s += '  \\draw [%s] %s --%s %s;\n' % (stylestr, pp[0], extra, pp[-1])
        return s

    def get_output_arrow_styles(self, arrow_style, edge):
        dot_arrow_head = edge.attr.get('arrowhead')
        dot_arrow_tail = edge.attr.get('arrowtail')
        output_arrow_style = arrow_style
        if dot_arrow_head:
            pgf_arrow_head = self.arrows_map_210.get(dot_arrow_head)
            if pgf_arrow_head:
                output_arrow_style = output_arrow_style.replace('>', pgf_arrow_head)
        if dot_arrow_tail:
            pgf_arrow_tail = self.arrows_map_210.get(dot_arrow_tail)
            if pgf_arrow_tail:
                output_arrow_style = output_arrow_style.replace('<', pgf_arrow_tail)
        return output_arrow_style

    def init_template_vars(self):
        DotConvBase.init_template_vars(self)
        if self.options.get('crop'):
            cropcode = '\\usepackage[active,tightpage]{preview}\n' + '\\PreviewEnvironment{tikzpicture}\n' + '\\setlength\\PreviewBorder{%s}' % self.options.get('margin', '0pt')
        else:
            cropcode = ''
        variables = {'<<cropcode>>': cropcode}
        self.templatevars.update(variables)

    def get_node_preproc_code(self, node):
        lblstyle = node.attr.get('lblstyle', '')
        text = node.attr.get('texlbl', '')
        if lblstyle:
            return '  \\tikz \\node[%s] {%s};\n' % (lblstyle, text)
        else:
            return '\\tikz \\node {' + text + '};'

    def get_edge_preproc_code(self, edge, attribute='texlbl'):
        lblstyle = edge.attr.get('lblstyle', '')
        text = edge.attr.get(attribute, '')
        if lblstyle:
            return '  \\tikz \\node[%s] {%s};\n' % (lblstyle, text)
        else:
            return '\\tikz \\node ' + '{' + text + '};'

    def get_graph_preproc_code(self, graph):
        lblstyle = graph.attr.get('lblstyle', '')
        text = graph.attr.get('texlbl', '')
        if lblstyle:
            return '  \\tikz \\node[%s] {%s};\n' % (lblstyle, text)
        else:
            return '\\tikz \\node {' + text + '};'