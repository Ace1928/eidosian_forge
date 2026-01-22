import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
class LinkEditor(PLinkBase):
    """
    A complete graphical link drawing tool based on the one embedded in Jeff Weeks'
    original SnapPea program.
    """

    def __init__(self, *args, **kwargs):
        if 'title' not in kwargs:
            kwargs['title'] = 'PLink Editor'
        self.callback = kwargs.pop('callback', None)
        self.cb_menu = kwargs.pop('cb_menu', '')
        self.no_arcs = kwargs.pop('no_arcs', False)
        PLinkBase.__init__(self, *args, **kwargs)
        self.flipcheck = None
        self.shift_down = False
        self.state = 'start_state'
        self.canvas.bind('<Button-1>', self.single_click)
        self.canvas.bind('<Double-Button-1>', self.double_click)
        self.canvas.bind('<Shift-Button-1>', self.shift_click)
        self.canvas.bind('<Motion>', self.mouse_moved)
        self.window.bind('<FocusIn>', self.focus_in)
        self.window.bind('<FocusOut>', self.focus_out)

    def _do_callback(self):
        if self._warn_arcs() == 'oops':
            return
        self.callback(self)

    def _check_update(self):
        if self.state == 'start_state':
            return True
        elif self.state == 'dragging_state':
            x, y = (self.cursorx, self.canvas.winfo_height() - self.cursory)
            self.write_text('(%d, %d)' % (x, y))
        return False

    def _add_file_menu(self):
        file_menu = Tk_.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label='Open File ...', command=self.load)
        file_menu.add_command(label='Save ...', command=self.save)
        self.build_save_image_menu(self.menubar, file_menu)
        file_menu.add_separator()
        if self.callback:
            file_menu.add_command(label='Close', command=self.done)
        else:
            file_menu.add_command(label='Quit', command=self.done)
        self.menubar.add_cascade(label='File', menu=file_menu)

    def _extend_style_menu(self, style_menu):
        style_menu.add_radiobutton(label='Smooth edit', value='both', command=self.set_style, variable=self.style_var)

    def _add_tools_menu(self):
        self.lock_var = Tk_.BooleanVar(self.window)
        self.lock_var.set(False)
        self.tools_menu = tools_menu = Tk_.Menu(self.menubar, tearoff=0)
        tools_menu.add_command(label='Make alternating', command=self.make_alternating)
        tools_menu.add_command(label='Reflect', command=self.reflect)
        tools_menu.add_checkbutton(label='Preserve diagram', var=self.lock_var)
        tools_menu.add_command(label='Clear', command=self.clear)
        if self.callback:
            tools_menu.add_command(label=self.cb_menu, command=self._do_callback)
        self.menubar.add_cascade(label='Tools', menu=tools_menu)

    def _key_release(self, event):
        """
        Handler for keyrelease events.
        """
        if not self.state == 'start_state':
            return
        if event.keysym in ('Shift_L', 'Shift_R'):
            self.shift_down = False
            self.set_start_cursor(self.cursorx, self.cursory)

    def _key_press(self, event):
        """
        Handler for keypress events.
        """
        dx, dy = (0, 0)
        key = event.keysym
        if key in ('Shift_L', 'Shift_R') and self.state == 'start_state':
            self.shift_down = True
            self.set_start_cursor(self.cursorx, self.cursory)
        if key in ('Delete', 'BackSpace'):
            if self.state == 'drawing_state':
                last_arrow = self.ActiveVertex.in_arrow
                if last_arrow:
                    dead_arrow = self.ActiveVertex.out_arrow
                    if dead_arrow:
                        self.destroy_arrow(dead_arrow)
                    self.ActiveVertex = last_arrow.start
                    self.ActiveVertex.out_arrow = None
                    x0, y0, x1, y1 = self.canvas.coords(self.LiveArrow1)
                    x0, y0 = self.ActiveVertex.point()
                    self.canvas.coords(self.LiveArrow1, x0, y0, x1, y1)
                    self.Crossings = [c for c in self.Crossings if last_arrow not in c]
                    self.Vertices.remove(last_arrow.end)
                    self.Arrows.remove(last_arrow)
                    last_arrow.end.erase()
                    last_arrow.erase()
                    for arrow in self.Arrows:
                        arrow.draw(self.Crossings)
                if not self.ActiveVertex.in_arrow:
                    self.Vertices.remove(self.ActiveVertex)
                    self.ActiveVertex.erase()
                    self.goto_start_state()
        elif key in ('plus', 'equal'):
            self.zoom_in()
        elif key in ('minus', 'underscore'):
            self.zoom_out()
        elif key == '0':
            self.zoom_to_fit()
        if self.state != 'dragging_state':
            try:
                self._shift(*canvas_shifts[key])
            except KeyError:
                pass
            return
        else:
            if key in ('Return', 'Escape'):
                self.cursorx = self.ActiveVertex.x
                self.cursory = self.ActiveVertex.y
                self.end_dragging_state()
                self.shifting = False
                return
            self._smooth_shift(key)
            return 'break'
        event.x, event.y = (self.cursorx, self.cursory)
        self.mouse_moved(event)

    def _warn_arcs(self):
        if self.no_arcs:
            for vertex in self.Vertices:
                if vertex.is_endpoint():
                    if tkMessageBox.askretrycancel('Warning', 'This link has non-closed components!\nClick "retry" to continue editing.\nClick "cancel" to quit anyway.\n(The link projection may be useless.)'):
                        return 'oops'
                    else:
                        break

    def done(self, event=None):
        if self._warn_arcs() == 'oops':
            return
        else:
            if self.focus_after:
                self.window.after_cancel(self.focus_after)
            self.window.destroy()

    def make_alternating(self):
        """
        Changes crossings to make the projection alternating.
        Requires that all components be closed.
        """
        try:
            crossing_components = self.crossing_components()
        except ValueError:
            tkMessageBox.showwarning('Error', 'Please close up all components first.')
            return
        need_flipping = set()
        for component in self.DT_code()[0]:
            need_flipping.update((c for c in component if c < 0))
        for crossing in self.Crossings:
            if crossing.hit2 in need_flipping or crossing.hit1 in need_flipping:
                crossing.reverse()
        self.clear_text()
        self.update_info()
        for arrow in self.Arrows:
            arrow.draw(self.Crossings)
        self.update_smooth()

    def reflect(self):
        for crossing in self.Crossings:
            crossing.reverse()
        self.clear_text()
        self.update_info()
        for arrow in self.Arrows:
            arrow.draw(self.Crossings)
        self.update_smooth()

    def clear(self):
        self.lock_var.set(False)
        for arrow in self.Arrows:
            arrow.erase()
        for vertex in self.Vertices:
            vertex.erase()
        self.canvas.delete('all')
        self.palette.reset()
        self.initialize(self.canvas)
        self.show_DT_var.set(0)
        self.show_labels_var.set(0)
        self.info_var.set(0)
        self.clear_text()
        self.goto_start_state()

    def focus_in(self, event):
        self.focus_after = self.window.after(100, self.notice_focus)

    def notice_focus(self):
        self.focus_after = None
        self.has_focus = True

    def focus_out(self, event):
        self.has_focus = False

    def shift_click(self, event):
        """
        Event handler for mouse shift-clicks.
        """
        if self.style_var.get() == 'smooth':
            return
        if self.lock_var.get():
            return
        if self.state == 'start_state':
            if not self.has_focus:
                return
        else:
            self.has_focus = True
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.clear_text()
        start_vertex = Vertex(x, y, self.canvas, style='hidden')
        if start_vertex in self.CrossPoints:
            crossing = self.Crossings[self.CrossPoints.index(start_vertex)]
            self.update_info()
            crossing.is_virtual = not crossing.is_virtual
            crossing.under.draw(self.Crossings)
            crossing.over.draw(self.Crossings)
            self.update_smooth()

    def single_click(self, event):
        """
        Event handler for mouse clicks.
        """
        if self.style_var.get() == 'smooth':
            return
        if self.state == 'start_state':
            if not self.has_focus:
                return
        else:
            self.has_focus = True
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.clear_text()
        start_vertex = Vertex(x, y, self.canvas, style='hidden')
        if self.state == 'start_state':
            if start_vertex in self.Vertices:
                self.state = 'dragging_state'
                self.hide_DT()
                self.hide_labels()
                self.update_info()
                self.canvas.config(cursor=closed_hand_cursor)
                self.ActiveVertex = self.Vertices[self.Vertices.index(start_vertex)]
                self.ActiveVertex.freeze()
                self.saved_crossing_data = self.active_crossing_data()
                x1, y1 = self.ActiveVertex.point()
                if self.ActiveVertex.in_arrow is None and self.ActiveVertex.out_arrow is None:
                    self.double_click(event)
                    return
                if self.ActiveVertex.in_arrow:
                    x0, y0 = self.ActiveVertex.in_arrow.start.point()
                    self.ActiveVertex.in_arrow.freeze()
                    self.LiveArrow1 = self.canvas.create_line(x0, y0, x1, y1, fill='red')
                if self.ActiveVertex.out_arrow:
                    x0, y0 = self.ActiveVertex.out_arrow.end.point()
                    self.ActiveVertex.out_arrow.freeze()
                    self.LiveArrow2 = self.canvas.create_line(x0, y0, x1, y1, fill='red')
                if self.lock_var.get():
                    self.attach_cursor('start')
                return
            elif self.lock_var.get():
                return
            elif start_vertex in self.CrossPoints:
                crossing = self.Crossings[self.CrossPoints.index(start_vertex)]
                if crossing.is_virtual:
                    crossing.is_virtual = False
                else:
                    crossing.reverse()
                self.update_info()
                crossing.under.draw(self.Crossings)
                crossing.over.draw(self.Crossings)
                self.update_smooth()
                return
            elif self.clicked_on_arrow(start_vertex):
                return
            elif not self.generic_vertex(start_vertex):
                start_vertex.erase()
                self.alert()
                return
            x1, y1 = start_vertex.point()
            start_vertex.set_color(self.palette.new())
            self.Vertices.append(start_vertex)
            self.ActiveVertex = start_vertex
            self.goto_drawing_state(x1, y1)
            return
        elif self.state == 'drawing_state':
            next_vertex = Vertex(x, y, self.canvas, style='hidden')
            if next_vertex == self.ActiveVertex:
                next_vertex.erase()
                dead_arrow = self.ActiveVertex.out_arrow
                if dead_arrow:
                    self.destroy_arrow(dead_arrow)
                self.goto_start_state()
                return
            if self.ActiveVertex.out_arrow:
                next_arrow = self.ActiveVertex.out_arrow
                next_arrow.set_end(next_vertex)
                next_vertex.in_arrow = next_arrow
                if not next_arrow.frozen:
                    next_arrow.hide()
            else:
                this_color = self.ActiveVertex.color
                next_arrow = Arrow(self.ActiveVertex, next_vertex, self.canvas, style='hidden', color=this_color)
                self.Arrows.append(next_arrow)
            next_vertex.set_color(next_arrow.color)
            if next_vertex in [v for v in self.Vertices if v.is_endpoint()]:
                if not self.generic_arrow(next_arrow):
                    self.alert()
                    return
                next_vertex.erase()
                next_vertex = self.Vertices[self.Vertices.index(next_vertex)]
                if next_vertex.in_arrow:
                    next_vertex.reverse_path()
                next_arrow.set_end(next_vertex)
                next_vertex.in_arrow = next_arrow
                if next_vertex.color != self.ActiveVertex.color:
                    self.palette.recycle(self.ActiveVertex.color)
                    next_vertex.recolor_incoming(color=next_vertex.color)
                self.update_crossings(next_arrow)
                next_arrow.expose(self.Crossings)
                self.goto_start_state()
                return
            if not (self.generic_vertex(next_vertex) and self.generic_arrow(next_arrow)):
                self.alert()
                self.destroy_arrow(next_arrow)
                return
            self.update_crossings(next_arrow)
            self.update_crosspoints()
            next_arrow.expose(self.Crossings)
            self.Vertices.append(next_vertex)
            next_vertex.expose()
            self.ActiveVertex = next_vertex
            self.canvas.coords(self.LiveArrow1, x, y, x, y)
            return
        elif self.state == 'dragging_state':
            try:
                self.end_dragging_state()
            except ValueError:
                self.alert()

    def double_click(self, event):
        """
        Event handler for mouse double-clicks.
        """
        if self.style_var.get() == 'smooth':
            return
        if self.lock_var.get():
            return
        x = x1 = self.canvas.canvasx(event.x)
        y = y1 = self.canvas.canvasy(event.y)
        self.clear_text()
        vertex = Vertex(x, y, self.canvas, style='hidden')
        if self.state == 'dragging_state':
            try:
                self.end_dragging_state()
            except ValueError:
                self.alert()
                return
            if vertex in [v for v in self.Vertices if v.is_endpoint()]:
                vertex.erase()
                vertex = self.Vertices[self.Vertices.index(vertex)]
                x0, y0 = x1, y1 = vertex.point()
                if vertex.out_arrow:
                    self.update_crosspoints()
                    vertex.reverse_path()
            elif vertex in self.Vertices:
                cut_vertex = self.Vertices[self.Vertices.index(vertex)]
                cut_vertex.recolor_incoming(palette=self.palette)
                cut_arrow = cut_vertex.in_arrow
                cut_vertex.in_arrow = None
                vertex = cut_arrow.start
                x1, y1 = cut_vertex.point()
                cut_arrow.freeze()
            self.ActiveVertex = vertex
            self.goto_drawing_state(x1, y1)
            return
        elif self.state == 'drawing_state':
            dead_arrow = self.ActiveVertex.out_arrow
            if dead_arrow:
                self.destroy_arrow(dead_arrow)
            self.goto_start_state()

    def set_start_cursor(self, x, y):
        point = Vertex(x, y, self.canvas, style='hidden')
        if self.shift_down:
            if point in self.CrossPoints:
                self.canvas.config(cursor='dot')
            else:
                self.canvas.config(cursor='')
        elif self.lock_var.get():
            if point in self.Vertices:
                self.flipcheck = None
                self.canvas.config(cursor=open_hand_cursor)
            else:
                self.canvas.config(cursor='')
        elif point in self.Vertices:
            self.flipcheck = None
            self.canvas.config(cursor=open_hand_cursor)
        elif point in self.CrossPoints:
            self.flipcheck = None
            self.canvas.config(cursor='exchange')
        elif self.cursor_on_arrow(point):
            now = time.time()
            if self.flipcheck is None:
                self.flipcheck = now
            elif now - self.flipcheck > 0.5:
                self.canvas.config(cursor='double_arrow')
        else:
            self.flipcheck = None
            self.canvas.config(cursor='')

    def mouse_moved(self, event):
        """
        Handler for mouse motion events.
        """
        if self.style_var.get() == 'smooth':
            return
        canvas = self.canvas
        X, Y = (event.x, event.y)
        x, y = (canvas.canvasx(X), canvas.canvasy(Y))
        self.cursorx, self.cursory = (X, Y)
        if self.state == 'start_state':
            self.set_start_cursor(x, y)
        elif self.state == 'drawing_state':
            x0, y0, x1, y1 = self.canvas.coords(self.LiveArrow1)
            self.canvas.coords(self.LiveArrow1, x0, y0, x, y)
        elif self.state == 'dragging_state':
            if self.shifting:
                self.window.event_generate('<Return>')
                return 'break'
            else:
                self.move_active(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def active_crossing_data(self):
        """
        Return the tuple of edges crossed by the in and out
        arrows of the active vertex.
        """
        assert self.ActiveVertex is not None
        active = self.ActiveVertex
        ignore = [active.in_arrow, active.out_arrow]
        return (self.crossed_arrows(active.in_arrow, ignore), self.crossed_arrows(active.out_arrow, ignore))

    def move_is_ok(self):
        return self.active_crossing_data() == self.saved_crossing_data

    def move_active(self, x, y):
        active = self.ActiveVertex
        if self.lock_var.get():
            x0, y0 = active.point()
            active.x, active.y = (float(x), float(y))
            if self.move_is_ok():
                if not self.generic_vertex(active):
                    active.x, active.y = (x0, y0)
                    if self.cursor_attached:
                        self.detach_cursor('non-generic active vertex')
                    self.canvas.delete('lock_error')
                    delta = 6
                    self.canvas.create_oval(x0 - delta, y0 - delta, x0 + delta, y0 + delta, outline='gray', fill=None, width=3, tags='lock_error')
                    return
                if not self.verify_drag():
                    active.x, active.y = (x0, y0)
                    if self.cursor_attached:
                        self.detach_cursor('non-generic diagram')
                    return
                if not self.cursor_attached:
                    self.attach_cursor('move is ok')
            else:
                if self.cursor_attached:
                    self.detach_cursor('bad move')
                active.x, active.y = (x0, y0)
                self.ActiveVertex.draw()
                return
            self.canvas.delete('lock_error')
        else:
            active.x, active.y = (float(x), float(y))
        self.ActiveVertex.draw()
        if self.LiveArrow1:
            x0, y0, x1, y1 = self.canvas.coords(self.LiveArrow1)
            self.canvas.coords(self.LiveArrow1, x0, y0, x, y)
        if self.LiveArrow2:
            x0, y0, x1, y1 = self.canvas.coords(self.LiveArrow2)
            self.canvas.coords(self.LiveArrow2, x0, y0, x, y)
        self.update_smooth()
        self.update_info()
        self.window.update_idletasks()

    def attach_cursor(self, reason=''):
        self.cursor_attached = True
        self.ActiveVertex.set_delta(8)

    def detach_cursor(self, reason=''):
        self.cursor_attached = False
        self.ActiveVertex.set_delta(2)

    def _smooth_shift(self, key):
        try:
            ddx, ddy = vertex_shifts[key]
        except KeyError:
            return
        self.shifting = True
        dx, dy = self.shift_delta
        dx += ddx
        dy += ddy
        now = time.time()
        if now - self.shift_stamp < 0.1:
            self.shift_delta = (dx, dy)
        else:
            self.cursorx = x = self.ActiveVertex.x + dx
            self.cursory = y = self.ActiveVertex.y + dy
            self.move_active(x, y)
            self.shift_delta = (0, 0)
            self.shift_stamp = now

    def clicked_on_arrow(self, vertex):
        for arrow in self.Arrows:
            if arrow.too_close(vertex):
                arrow.end.reverse_path(self.Crossings)
                self.update_info()
                return True
        return False

    def cursor_on_arrow(self, point):
        if self.lock_var.get():
            return False
        for arrow in self.Arrows:
            if arrow.too_close(point):
                return True
        return False

    def goto_start_state(self):
        self.canvas.delete('lock_error')
        self.canvas.delete(self.LiveArrow1)
        self.LiveArrow1 = None
        self.canvas.delete(self.LiveArrow2)
        self.LiveArrow2 = None
        self.ActiveVertex = None
        self.update_crosspoints()
        self.state = 'start_state'
        self.set_style()
        self.update_info()
        self.canvas.config(cursor='')

    def goto_drawing_state(self, x1, y1):
        self.ActiveVertex.expose()
        self.ActiveVertex.draw()
        x0, y0 = self.ActiveVertex.point()
        self.LiveArrow1 = self.canvas.create_line(x0, y0, x1, y1, fill='red')
        self.state = 'drawing_state'
        self.canvas.config(cursor='pencil')
        self.hide_DT()
        self.hide_labels()
        self.clear_text()

    def verify_drag(self):
        active = self.ActiveVertex
        active.update_arrows()
        self.update_crossings(active.in_arrow)
        self.update_crossings(active.out_arrow)
        self.update_crosspoints()
        return self.generic_arrow(active.in_arrow) and self.generic_arrow(active.out_arrow)

    def end_dragging_state(self):
        if not self.verify_drag():
            raise ValueError
        if self.lock_var.get():
            self.detach_cursor()
            self.saved_crossing_data = None
        else:
            x, y = (float(self.cursorx), float(self.cursory))
            self.ActiveVertex.x, self.ActiveVertex.y = (x, y)
        endpoint = None
        if self.ActiveVertex.is_endpoint():
            other_ends = [v for v in self.Vertices if v.is_endpoint() and v is not self.ActiveVertex]
            if self.ActiveVertex in other_ends:
                endpoint = other_ends[other_ends.index(self.ActiveVertex)]
                self.ActiveVertex.swallow(endpoint, self.palette)
                self.Vertices = [v for v in self.Vertices if v is not endpoint]
            self.update_crossings(self.ActiveVertex.in_arrow)
            self.update_crossings(self.ActiveVertex.out_arrow)
        if endpoint is None and (not self.generic_vertex(self.ActiveVertex)):
            raise ValueError
        self.ActiveVertex.expose()
        if self.style_var.get() != 'smooth':
            if self.ActiveVertex.in_arrow:
                self.ActiveVertex.in_arrow.expose()
            if self.ActiveVertex.out_arrow:
                self.ActiveVertex.out_arrow.expose()
        self.goto_start_state()

    def generic_vertex(self, vertex):
        if vertex in [v for v in self.Vertices if v is not vertex]:
            return False
        for arrow in self.Arrows:
            if arrow.too_close(vertex, tolerance=Arrow.epsilon + 2):
                return False
        return True

    def generic_arrow(self, arrow):
        if arrow == None:
            return True
        locked = self.lock_var.get()
        for vertex in self.Vertices:
            if arrow.too_close(vertex):
                if locked:
                    x, y, delta = (vertex.x, vertex.y, 6)
                    self.canvas.delete('lock_error')
                    self.canvas.create_oval(x - delta, y - delta, x + delta, y + delta, outline='gray', fill=None, width=3, tags='lock_error')
                return False
        for crossing in self.Crossings:
            point = self.CrossPoints[self.Crossings.index(crossing)]
            if arrow not in crossing and arrow.too_close(point):
                if locked:
                    x, y, delta = (point.x, point.y, 6)
                    self.canvas.delete('lock_error')
                    self.canvas.create_oval(x - delta, y - delta, x + delta, y + delta, outline='gray', fill=None, width=3, tags='lock_error')
                return False
        return True

    def destroy_arrow(self, arrow):
        self.Arrows.remove(arrow)
        if arrow.end:
            arrow.end.in_arrow = None
        if arrow.start:
            arrow.start.out_arrow = None
        arrow.erase()
        self.Crossings = [c for c in self.Crossings if arrow not in c]

    def update_crossings(self, this_arrow):
        """
        Redraw any arrows which were changed by moving this_arrow.
        """
        if this_arrow == None:
            return
        cross_list = [c for c in self.Crossings if this_arrow in c]
        damage_list = []
        find = lambda x: cross_list[cross_list.index(x)]
        for arrow in self.Arrows:
            if this_arrow == arrow:
                continue
            new_crossing = Crossing(this_arrow, arrow)
            new_crossing.locate()
            if new_crossing.x != None:
                if new_crossing in cross_list:
                    find(new_crossing).locate()
                    continue
                else:
                    self.Crossings.append(new_crossing)
            elif new_crossing in self.Crossings:
                if arrow == find(new_crossing).under:
                    damage_list.append(arrow)
                self.Crossings.remove(new_crossing)
        for arrow in damage_list:
            arrow.draw(self.Crossings)

    def crossed_arrows(self, arrow, ignore_list=[]):
        """
        Return a tuple containing the arrows of the diagram which are
        crossed by the given arrow, in order along the given arrow.
        """
        if arrow is None:
            return tuple()
        arrow.vectorize()
        crosslist = []
        for n, diagram_arrow in enumerate(self.Arrows):
            if arrow == diagram_arrow or diagram_arrow in ignore_list:
                continue
            t = arrow ^ diagram_arrow
            if t is not None:
                crosslist.append((t, n))
        return tuple((n for _, n in sorted(crosslist)))