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
class PLinkBase(LinkViewer):
    """
    Base class for windows displaying a LinkViewer and an Info Window.
    """

    def __init__(self, root=None, manifold=None, file_name=None, title='', show_crossing_labels=False):
        self.initialize()
        self.show_crossing_labels = show_crossing_labels
        self.manifold = manifold
        self.title = title
        self.cursorx = 0
        self.cursory = 0
        self.colors = []
        self.color_keys = []
        if root is None:
            self.window = root = IPythonTkRoot(className='plink')
        else:
            self.window = Tk_.Toplevel(root)
        self.window.protocol('WM_DELETE_WINDOW', self.done)
        if sys.platform == 'linux2' or sys.platform == 'linux':
            root.tk.call('namespace', 'import', '::tk::dialog::file::')
            root.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
            root.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
        self.window.title(title)
        self.style = PLinkStyle()
        self.palette = Palette()
        self.frame = ttk.Frame(self.window)
        self.canvas = Tk_.Canvas(self.frame, bg='#dcecff', width=500, height=500, borderwidth=0, highlightthickness=0)
        self.smoother = smooth.Smoother(self.canvas)
        self.infoframe = ttk.Frame(self.window)
        self.infotext_contents = Tk_.StringVar(self.window)
        self.infotext = ttk.Entry(self.infoframe, state='readonly', font='Helvetica 14', textvariable=self.infotext_contents)
        self.infoframe.pack(padx=0, pady=0, fill=Tk_.X, expand=Tk_.NO, side=Tk_.BOTTOM)
        self.frame.pack(padx=0, pady=0, fill=Tk_.BOTH, expand=Tk_.YES)
        self.canvas.pack(padx=0, pady=0, fill=Tk_.BOTH, expand=Tk_.YES)
        self.infotext.pack(padx=5, pady=0, fill=Tk_.X, expand=Tk_.YES)
        self.show_DT_var = Tk_.IntVar(self.window)
        self.show_labels_var = Tk_.IntVar(self.window)
        self.info_var = Tk_.IntVar(self.window)
        self.style_var = Tk_.StringVar(self.window)
        self.style_var.set('pl')
        self.cursor_attached = False
        self.saved_crossing_data = None
        self.current_info = 0
        self.has_focus = True
        self.focus_after = None
        self.infotext.bind('<Control-Shift-C>', lambda event: self.infotext.event_generate('<<Copy>>'))
        self.infotext.bind('<<Copy>>', self.copy_info)
        self._build_menus()
        self.window.bind('<Key>', self._key_press)
        self.window.bind('<KeyRelease>', self._key_release)
        if file_name:
            self.load(file_name=file_name)

    def _key_release(self, event):
        """
        Handler for keyrelease events.
        """
        pass

    def _key_press(self, event):
        """
        Handler for keypress events.
        """
        dx, dy = (0, 0)
        key = event.keysym
        if key in ('plus', 'equal'):
            self.zoom_in()
        elif key in ('minus', 'underscore'):
            self.zoom_out()
        elif key == '0':
            self.zoom_to_fit()
        try:
            self._shift(*canvas_shifts[key])
        except KeyError:
            pass
        return

    def _build_menus(self):
        self.menubar = menubar = Tk_.Menu(self.window)
        self._add_file_menu()
        self._add_info_menu()
        self._add_tools_menu()
        self._add_style_menu()
        self.window.config(menu=menubar)
        help_menu = Tk_.Menu(menubar, tearoff=0)
        help_menu.add_command(label='About PLink...', command=self.about)
        help_menu.add_command(label='Instructions ...', command=self.howto)
        menubar.add_cascade(label='Help', menu=help_menu)

    def _add_file_menu(self):
        file_menu = Tk_.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label='Save ...', command=self.save)
        self.build_save_image_menu(self.menubar, file_menu)
        file_menu.add_separator()
        file_menu.add_command(label='Quit', command=self.done)
        self.menubar.add_cascade(label='File', menu=file_menu)

    def _add_info_menu(self):
        info_menu = Tk_.Menu(self.menubar, tearoff=0)
        info_menu.add_radiobutton(label='DT code', var=self.info_var, command=self.set_info, value=1)
        info_menu.add_radiobutton(label='Alphabetical DT', var=self.info_var, command=self.set_info, value=2)
        info_menu.add_radiobutton(label='Gauss code', var=self.info_var, command=self.set_info, value=3)
        info_menu.add_radiobutton(label='PD code', var=self.info_var, command=self.set_info, value=4)
        info_menu.add_radiobutton(label='BB framing', var=self.info_var, command=self.set_info, value=5)
        info_menu.add_separator()
        info_menu.add_checkbutton(label='DT labels', var=self.show_DT_var, command=self.update_info)
        if self.show_crossing_labels:
            info_menu.add_checkbutton(label='Crossing labels', var=self.show_labels_var, command=self.update_info)
        self.menubar.add_cascade(label='Info', menu=info_menu)

    def _add_tools_menu(self):
        pass

    def _add_style_menu(self):
        style_menu = Tk_.Menu(self.menubar, tearoff=0)
        style_menu.add_radiobutton(label='PL', value='pl', command=self.set_style, variable=self.style_var)
        style_menu.add_radiobutton(label='Smooth', value='smooth', command=self.set_style, variable=self.style_var)
        self._extend_style_menu(style_menu)
        self.menubar.add_cascade(label='Style', menu=style_menu)
        self._add_zoom_and_pan(style_menu)

    def _extend_style_menu(self, style_menu):
        pass

    def _add_zoom_and_pan(self, style_menu):
        zoom_menu = Tk_.Menu(style_menu, tearoff=0)
        pan_menu = Tk_.Menu(style_menu, tearoff=0)
        if sys.platform == 'darwin':
            zoom_menu.add_command(label='Zoom in    \t+', command=self.zoom_in)
            zoom_menu.add_command(label='Zoom out   \t-', command=self.zoom_out)
            zoom_menu.add_command(label='Zoom to fit\t0', command=self.zoom_to_fit)
            pan_menu.add_command(label='Left  \t' + scut['Left'], command=lambda: self._shift(-5, 0))
            pan_menu.add_command(label='Up    \t' + scut['Up'], command=lambda: self._shift(0, -5))
            pan_menu.add_command(label='Right \t' + scut['Right'], command=lambda: self._shift(5, 0))
            pan_menu.add_command(label='Down  \t' + scut['Down'], command=lambda: self._shift(0, 5))
        else:
            zoom_menu.add_command(label='Zoom in', accelerator='+', command=self.zoom_in)
            zoom_menu.add_command(label='Zoom out', accelerator='-', command=self.zoom_out)
            zoom_menu.add_command(label='Zoom to fit', accelerator='0', command=self.zoom_to_fit)
            pan_menu.add_command(label='Left', accelerator=scut['Left'], command=lambda: self._shift(-5, 0))
            pan_menu.add_command(label='Up', accelerator=scut['Up'], command=lambda: self._shift(0, -5))
            pan_menu.add_command(label='Right', accelerator=scut['Right'], command=lambda: self._shift(5, 0))
            pan_menu.add_command(label='Down', accelerator=scut['Down'], command=lambda: self._shift(0, 5))
        style_menu.add_separator()
        style_menu.add_cascade(label='Zoom', menu=zoom_menu)
        style_menu.add_cascade(label='Pan', menu=pan_menu)

    def alert(self):
        background = self.canvas.cget('bg')

        def reset_bg():
            self.canvas.config(bg=background)
        self.canvas.config(bg='#000000')
        self.canvas.after(100, reset_bg)

    def done(self, event=None):
        self.window.destroy()

    def reopen(self):
        try:
            self.window.deiconify()
        except Tk_.TclError:
            print('The PLink window was destroyed')

    def set_style(self):
        mode = self.style_var.get()
        if mode == 'smooth':
            self.canvas.config(background='#ffffff')
            self.enable_fancy_save_images()
            for vertex in self.Vertices:
                vertex.hide()
            for arrow in self.Arrows:
                arrow.hide()
        elif mode == 'both':
            self.canvas.config(background='#ffffff')
            self.disable_fancy_save_images()
            for vertex in self.Vertices:
                vertex.expose()
            for arrow in self.Arrows:
                arrow.make_faint()
        else:
            self.canvas.config(background='#dcecff')
            self.enable_fancy_save_images()
            for vertex in self.Vertices:
                vertex.expose()
            for arrow in self.Arrows:
                arrow.expose()
        self.full_redraw()

    def full_redraw(self):
        """
        Recolors and redraws all components, in DT order, and displays
        the legend linking colors to cusp indices.
        """
        components = self.arrow_components(include_isolated_vertices=True)
        self.colors = []
        for key in self.color_keys:
            self.canvas.delete(key)
        self.color_keys = []
        x, y, n = (10, 5, 0)
        self.palette.reset()
        for component in components:
            color = self.palette.new()
            self.colors.append(color)
            component[0].start.color = color
            for arrow in component:
                arrow.color = color
                arrow.end.color = color
                arrow.draw(self.Crossings)
            if self.style_var.get() != 'smooth':
                self.color_keys.append(self.canvas.create_text(x, y, text=str(n), fill=color, anchor=Tk_.NW, font='Helvetica 16 bold'))
            x, n = (x + 16, n + 1)
        for vertex in self.Vertices:
            vertex.draw()
        self.update_smooth()

    def unpickle(self, vertices, arrows, crossings, hot=None):
        LinkManager.unpickle(self, vertices, arrows, crossings, hot)
        self.set_style()
        self.full_redraw()

    def set_info(self):
        self.clear_text()
        which_info = self.info_var.get()
        if which_info == self.current_info:
            self.info_var.set(0)
            self.current_info = 0
        else:
            self.current_info = which_info
            self.update_info()

    def copy_info(self, event):
        self.window.clipboard_clear()
        if self.infotext.selection_present():
            self.window.clipboard_append(self.infotext.selection_get())
            self.infotext.selection_clear()

    def clear_text(self):
        self.infotext_contents.set('')
        self.window.focus_set()

    def write_text(self, string):
        self.infotext_contents.set(string)

    def _shift(self, dx, dy):
        for vertex in self.Vertices:
            vertex.x += dx
            vertex.y += dy
        self.canvas.move('transformable', dx, dy)
        for livearrow in (self.LiveArrow1, self.LiveArrow2):
            if livearrow:
                x0, y0, x1, y1 = self.canvas.coords(livearrow)
                x0 += dx
                y0 += dy
                self.canvas.coords(livearrow, x0, y0, x1, y1)

    def _zoom(self, xfactor, yfactor):
        try:
            ulx, uly, lrx, lry = self.canvas.bbox('transformable')
        except TypeError:
            return
        for vertex in self.Vertices:
            vertex.x = ulx + xfactor * (vertex.x - ulx)
            vertex.y = uly + yfactor * (vertex.y - uly)
        self.update_crosspoints()
        for arrow in self.Arrows:
            arrow.draw(self.Crossings, skip_frozen=False)
        for vertex in self.Vertices:
            vertex.draw(skip_frozen=False)
        self.update_smooth()
        for livearrow in (self.LiveArrow1, self.LiveArrow2):
            if livearrow:
                x0, y0, x1, y1 = self.canvas.coords(livearrow)
                x0 = ulx + xfactor * (x0 - ulx)
                y0 = uly + yfactor * (y0 - uly)
                self.canvas.coords(livearrow, x0, y0, x1, y1)
        self.update_info()

    def zoom_in(self):
        self._zoom(1.2, 1.2)

    def zoom_out(self):
        self._zoom(0.8, 0.8)

    def zoom_to_fit(self):
        W, H = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if W < 10:
            W, H = (self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight())
        x0, y0, x1, y1 = (W, H, 0, 0)
        for V in self.Vertices:
            x0, y0 = (min(x0, V.x), min(y0, V.y))
            x1, y1 = (max(x1, V.x), max(y1, V.y))
        w, h = (x1 - x0, y1 - y0)
        factor = min((W - 60) / w, (H - 60) / h)
        xfactor, yfactor = (round(factor * w) / w, round(factor * h) / h)
        self._zoom(xfactor, yfactor)
        try:
            x0, y0, x1, y1 = self.canvas.bbox('transformable')
            self._shift((W - x1 + x0) / 2 - x0, (H - y1 + y0) / 2 - y0)
        except TypeError:
            pass

    def update_smooth(self):
        self.smoother.clear()
        mode = self.style_var.get()
        if mode == 'smooth':
            self.smoother.set_polylines(self.polylines())
        elif mode == 'both':
            self.smoother.set_polylines(self.polylines(), thickness=2)

    def _check_update(self):
        return True

    def update_info(self):
        self.hide_DT()
        self.hide_labels()
        self.clear_text()
        if not self._check_update():
            return
        if self.show_DT_var.get():
            dt = self.DT_code()
            if dt is not None:
                self.show_DT()
        if self.show_labels_var.get():
            self.show_labels()
        info_value = self.info_var.get()
        if info_value == 1:
            self.DT_normal()
        elif info_value == 2:
            self.DT_alpha()
        elif info_value == 3:
            self.Gauss_info()
        elif info_value == 4:
            self.PD_info()
        elif info_value == 5:
            self.BB_info()

    def show_labels(self):
        """
        Display the assigned labels next to each crossing.
        """
        for crossing in self.Crossings:
            crossing.locate()
            yshift = 0
            for arrow in (crossing.over, crossing.under):
                arrow.vectorize()
                if abs(arrow.dy) < 0.3 * abs(arrow.dx):
                    yshift = 8
            flip = ' *' if crossing.flipped else ''
            self.labels.append(self.canvas.create_text((crossing.x - 1, crossing.y - yshift), anchor=Tk_.E, text=str(crossing.label)))

    def show_DT(self):
        """
        Display the DT hit counters next to each crossing.  Crossings
        that need to be flipped for the planar embedding have an
        asterisk.
        """
        for crossing in self.Crossings:
            crossing.locate()
            yshift = 0
            for arrow in (crossing.over, crossing.under):
                arrow.vectorize()
                if abs(arrow.dy) < 0.3 * abs(arrow.dx):
                    yshift = 8
            flip = ' *' if crossing.flipped else ''
            self.DTlabels.append(self.canvas.create_text((crossing.x - 10, crossing.y - yshift), anchor=Tk_.E, text=str(crossing.hit1)))
            self.DTlabels.append(self.canvas.create_text((crossing.x + 10, crossing.y - yshift), anchor=Tk_.W, text=str(crossing.hit2) + flip))

    def hide_labels(self):
        for text_item in self.labels:
            self.canvas.delete(text_item)
        self.labels = []

    def hide_DT(self):
        for text_item in self.DTlabels:
            self.canvas.delete(text_item)
        self.DTlabels = []

    def not_done(self):
        tkMessageBox.showwarning('Not implemented', 'Sorry!  That feature has not been written yet.')

    def load(self, file_name=None):
        if file_name:
            loadfile = open(file_name, 'r')
        else:
            loadfile = askopenfile(parent=self.window)
        if loadfile:
            contents = loadfile.read()
            loadfile.close()
            self.clear()
            self.clear_text()
            hot = self._from_string(contents)
            self.window.update()
            if hot:
                self.ActiveVertex = self.Vertices[hot]
                self.goto_drawing_state(*self.canvas.winfo_pointerxy())
            else:
                self.zoom_to_fit()
                self.goto_start_state()

    def save(self):
        savefile = asksaveasfile(parent=self.window, mode='w', title='Save As Snappea Projection File', defaultextension='.lnk', filetypes=[('Link and text files', '*.lnk *.txt', 'TEXT'), ('All text files', '', 'TEXT'), ('All files', '')])
        if savefile:
            savefile.write(self.SnapPea_projection_file())
            savefile.close()

    def save_image(self, file_type='eps', colormode='color'):
        mode = self.style_var.get()
        target = self.smoother if mode == 'smooth' else self
        LinkViewer.save_image(self, file_type, colormode, target)

    def about(self):
        InfoDialog(self.window, 'About PLink', self.style, About)

    def howto(self):
        doc_file = os.path.join(os.path.dirname(__file__), 'doc', 'index.html')
        doc_path = os.path.abspath(doc_file)
        url = 'file:' + pathname2url(doc_path)
        try:
            webbrowser.open(url)
        except:
            tkMessageBox.showwarning('Not found!', 'Could not open URL\n(%s)' % url)