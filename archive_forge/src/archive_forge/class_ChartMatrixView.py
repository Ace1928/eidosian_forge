import os.path
import pickle
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from tkinter.messagebox import showerror, showinfo
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal
from nltk.parse.chart import (
from nltk.tree import Tree
from nltk.util import in_idle
class ChartMatrixView:
    """
    A view of a chart that displays the contents of the corresponding matrix.
    """

    def __init__(self, parent, chart, toplevel=True, title='Chart Matrix', show_numedges=False):
        self._chart = chart
        self._cells = []
        self._marks = []
        self._selected_cell = None
        if toplevel:
            self._root = Toplevel(parent)
            self._root.title(title)
            self._root.bind('<Control-q>', self.destroy)
            self._init_quit(self._root)
        else:
            self._root = Frame(parent)
        self._init_matrix(self._root)
        self._init_list(self._root)
        if show_numedges:
            self._init_numedges(self._root)
        else:
            self._numedges_label = None
        self._callbacks = {}
        self._num_edges = 0
        self.draw()

    def _init_quit(self, root):
        quit = Button(root, text='Quit', command=self.destroy)
        quit.pack(side='bottom', expand=0, fill='none')

    def _init_matrix(self, root):
        cframe = Frame(root, border=2, relief='sunken')
        cframe.pack(expand=0, fill='none', padx=1, pady=3, side='top')
        self._canvas = Canvas(cframe, width=200, height=200, background='white')
        self._canvas.pack(expand=0, fill='none')

    def _init_numedges(self, root):
        self._numedges_label = Label(root, text='0 edges')
        self._numedges_label.pack(expand=0, fill='none', side='top')

    def _init_list(self, root):
        self._list = EdgeList(root, [], width=20, height=5)
        self._list.pack(side='top', expand=1, fill='both', pady=3)

        def cb(edge, self=self):
            self._fire_callbacks('select', edge)
        self._list.add_callback('select', cb)
        self._list.focus()

    def destroy(self, *e):
        if self._root is None:
            return
        try:
            self._root.destroy()
        except:
            pass
        self._root = None

    def set_chart(self, chart):
        if chart is not self._chart:
            self._chart = chart
            self._num_edges = 0
            self.draw()

    def update(self):
        if self._root is None:
            return
        N = len(self._cells)
        cell_edges = [[0 for i in range(N)] for j in range(N)]
        for edge in self._chart:
            cell_edges[edge.start()][edge.end()] += 1
        for i in range(N):
            for j in range(i, N):
                if cell_edges[i][j] == 0:
                    color = 'gray20'
                else:
                    color = '#00{:02x}{:02x}'.format(min(255, 50 + 128 * cell_edges[i][j] / 10), max(0, 128 - 128 * cell_edges[i][j] / 10))
                cell_tag = self._cells[i][j]
                self._canvas.itemconfig(cell_tag, fill=color)
                if (i, j) == self._selected_cell:
                    self._canvas.itemconfig(cell_tag, outline='#00ffff', width=3)
                    self._canvas.tag_raise(cell_tag)
                else:
                    self._canvas.itemconfig(cell_tag, outline='black', width=1)
        edges = list(self._chart.select(span=self._selected_cell))
        self._list.set(edges)
        self._num_edges = self._chart.num_edges()
        if self._numedges_label is not None:
            self._numedges_label['text'] = '%d edges' % self._num_edges

    def activate(self):
        self._canvas.itemconfig('inactivebox', state='hidden')
        self.update()

    def inactivate(self):
        self._canvas.itemconfig('inactivebox', state='normal')
        self.update()

    def add_callback(self, event, func):
        self._callbacks.setdefault(event, {})[func] = 1

    def remove_callback(self, event, func=None):
        if func is None:
            del self._callbacks[event]
        else:
            try:
                del self._callbacks[event][func]
            except:
                pass

    def _fire_callbacks(self, event, *args):
        if event not in self._callbacks:
            return
        for cb_func in list(self._callbacks[event].keys()):
            cb_func(*args)

    def select_cell(self, i, j):
        if self._root is None:
            return
        if (i, j) == self._selected_cell and self._chart.num_edges() == self._num_edges:
            return
        self._selected_cell = (i, j)
        self.update()
        self._fire_callbacks('select_cell', i, j)

    def deselect_cell(self):
        if self._root is None:
            return
        self._selected_cell = None
        self._list.set([])
        self.update()

    def _click_cell(self, i, j):
        if self._selected_cell == (i, j):
            self.deselect_cell()
        else:
            self.select_cell(i, j)

    def view_edge(self, edge):
        self.select_cell(*edge.span())
        self._list.view(edge)

    def mark_edge(self, edge):
        if self._root is None:
            return
        self.select_cell(*edge.span())
        self._list.mark(edge)

    def unmark_edge(self, edge=None):
        if self._root is None:
            return
        self._list.unmark(edge)

    def markonly_edge(self, edge):
        if self._root is None:
            return
        self.select_cell(*edge.span())
        self._list.markonly(edge)

    def draw(self):
        if self._root is None:
            return
        LEFT_MARGIN = BOT_MARGIN = 15
        TOP_MARGIN = 5
        c = self._canvas
        c.delete('all')
        N = self._chart.num_leaves() + 1
        dx = (int(c['width']) - LEFT_MARGIN) / N
        dy = (int(c['height']) - TOP_MARGIN - BOT_MARGIN) / N
        c.delete('all')
        for i in range(N):
            c.create_text(LEFT_MARGIN - 2, i * dy + dy / 2 + TOP_MARGIN, text=repr(i), anchor='e')
            c.create_text(i * dx + dx / 2 + LEFT_MARGIN, N * dy + TOP_MARGIN + 1, text=repr(i), anchor='n')
            c.create_line(LEFT_MARGIN, dy * (i + 1) + TOP_MARGIN, dx * N + LEFT_MARGIN, dy * (i + 1) + TOP_MARGIN, dash='.')
            c.create_line(dx * i + LEFT_MARGIN, TOP_MARGIN, dx * i + LEFT_MARGIN, dy * N + TOP_MARGIN, dash='.')
        c.create_rectangle(LEFT_MARGIN, TOP_MARGIN, LEFT_MARGIN + dx * N, dy * N + TOP_MARGIN, width=2)
        self._cells = [[None for i in range(N)] for j in range(N)]
        for i in range(N):
            for j in range(i, N):
                t = c.create_rectangle(j * dx + LEFT_MARGIN, i * dy + TOP_MARGIN, (j + 1) * dx + LEFT_MARGIN, (i + 1) * dy + TOP_MARGIN, fill='gray20')
                self._cells[i][j] = t

                def cb(event, self=self, i=i, j=j):
                    self._click_cell(i, j)
                c.tag_bind(t, '<Button-1>', cb)
        xmax, ymax = (int(c['width']), int(c['height']))
        t = c.create_rectangle(-100, -100, xmax + 100, ymax + 100, fill='gray50', state='hidden', tag='inactivebox')
        c.tag_lower(t)
        self.update()

    def pack(self, *args, **kwargs):
        self._root.pack(*args, **kwargs)