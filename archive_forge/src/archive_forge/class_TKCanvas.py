import tkFont
import Tkinter
import rdkit.sping.pid
class TKCanvas(tk.Canvas, rdkit.sping.pid.Canvas):
    __TRANSPARENT = ''

    def __init__(self, size=(300, 300), name='sping.TK', master=None, scrollingViewPortSize=None, **kw):
        """This canvas allows you to add a tk.Canvas with a sping API for drawing.
        To add scrollbars, the simpliest method is to set the 'scrollingViewPortSize'
        equal to a tuple that describes the width and height of the visible porition
        of the canvas on screen.  This sets scrollregion=(0,0, size[0], size[1]).
        Then you can add scrollbars as you would any tk.Canvas.

        Note, because this is a subclass of tk.Canvas, you can use the normal keywords
        to specify a tk.Canvas with scrollbars, however, you should then be careful to
        set the "scrollregion" option to the same size as the 'size' passed to __init__.
        Tkinter's scrollregion option essentially makes 'size' ignored.  """
        rdkit.sping.pid.Canvas.__init__(self, size=size, name=size)
        if scrollingViewPortSize:
            kw['scrollregion'] = (0, 0, size[0], size[1])
            kw['height'] = scrollingViewPortSize[0]
            kw['width'] = scrollingViewPortSize[1]
        else:
            kw['width'] = size[0]
            kw['height'] = size[1]
        tk.Canvas.__init__(self, master, **kw)
        self.config(background='white')
        self.width, self.height = size
        self._font_manager = FontManager(self)
        self._configure()
        self._item_ids = []
        self._images = []

    def _configure(self):
        pass

    def _display(self):
        self.flush()
        self.mainloop()

    def _quit(self):
        self.quit()

    def _to_ps_file(self, filename):
        self.postscript(file=filename)

    def isInteractive(self):
        return 0

    def onOver(self, event):
        pass

    def onClick(self, event):
        pass

    def onKey(self, event):
        pass

    def flush(self):
        tk.Canvas.update(self)

    def clear(self):
        map(self.delete, self._item_ids)
        self._item_ids = []

    def _colorToTkColor(self, c):
        return '#%02X%02X%02X' % (int(c.red * 255), int(c.green * 255), int(c.blue * 255))

    def _getTkColor(self, color, defaultColor):
        if color is None:
            color = defaultColor
        if color is rdkit.sping.pid.transparent:
            color = self.__TRANSPARENT
        else:
            color = self._colorToTkColor(color)
        return color

    def drawLine(self, x1, y1, x2, y2, color=None, width=None):
        color = self._getTkColor(color, self.defaultLineColor)
        if width is None:
            width = self.defaultLineWidth
        new_item = self.create_line(x1, y1, x2, y2, fill=color, width=width)
        self._item_ids.append(new_item)

    def stringWidth(self, s, font=None):
        return self._font_manager.stringWidth(s, font or self.defaultFont)

    def fontAscent(self, font=None):
        return self._font_manager.fontAscent(font or self.defaultFont)

    def fontDescent(self, font=None):
        return self._font_manager.fontDescent(font or self.defaultFont)

    def drawString(self, s, x, y, font=None, color=None, angle=None):
        if angle:
            try:
                self._drawRotatedString(s, x, y, font, color, angle)
                return
            except ImportError:
                print('PIL not available. Using unrotated strings.')
        y = y - self.fontHeight(font) * 0.28
        color = self._getTkColor(color, self.defaultLineColor)
        font = self._font_manager.getTkFontString(font or self.defaultFont)
        new_item = self.create_text(x, y, text=s, font=font, fill=color, anchor=Tkinter.W)
        self._item_ids.append(new_item)

    def _drawRotatedString(self, s, x, y, font=None, color=None, angle=0):
        try:
            from PIL import Image, ImageTk
            import rdkit.sping.PIL.pidPIL
            pp = rdkit.sping.PIL.pidPIL
        except ImportError:
            raise ImportError('Rotated strings only possible with PIL support')
        pilCan = pp.PILCanvas(size=(self.width, self.height))
        pilCan.defaultFont = self.defaultFont
        pilCan.defaultLineColor = self.defaultLineColor
        if '\n' in s or '\r' in s:
            self.drawMultiLineString(s, x, y, font, color, angle)
            return
        if not font:
            font = pilCan.defaultFont
        if not color:
            color = self.defaultLineColor
        if color == rdkit.sping.pid.transparent:
            return
        tempsize = pilCan.stringWidth(s, font) * 1.2
        tempimg = Image.new('RGB', (tempsize, tempsize), (0, 0, 0))
        txtimg = Image.new('RGB', (tempsize, tempsize), (255, 255, 255))
        from PIL import ImageDraw
        temppen = ImageDraw.ImageDraw(tempimg)
        temppen.setink((255, 255, 255))
        pilfont = pp._pilFont(font)
        if not pilfont:
            raise ValueError('Bad font: %s' % font)
        temppen.setfont(pilfont)
        pos = [4, int(tempsize / 2 - pilCan.fontAscent(font)) - pilCan.fontDescent(font)]
        temppen.text(pos, s)
        pos[1] = int(tempsize / 2)
        if angle:
            from math import cos, pi, sin
            tempimg = tempimg.rotate(angle, Image.BILINEAR)
            temppen = ImageDraw.ImageDraw(tempimg)
            radians = -angle * pi / 180.0
            r = tempsize / 2 - pos[0]
            pos[0] = int(tempsize / 2 - r * cos(radians))
            pos[1] = int(pos[1] - r * sin(radians))
        mask = tempimg.convert('L').point(lambda c: c)
        temppen.setink((color.red * 255, color.green * 255, color.blue * 255))
        temppen.setfill(1)
        temppen.rectangle((0, 0, tempsize, tempsize))
        txtimg.paste(tempimg, (0, 0), mask)
        transp = txtimg.convert('RGBA')
        source = transp.split()
        R, G, B, A = (0, 1, 2, 3)
        mask = transp.point(lambda i: i < 255 and 255)
        source[A].paste(mask)
        transp = Image.merge(transp.mode, source)
        self.drawImage(transp, x - pos[0], y - pos[1])

    def drawRect(self, x1, y1, x2, y2, edgeColor=None, edgeWidth=None, fillColor=None):
        fillColor = self._getTkColor(fillColor, self.defaultFillColor)
        edgeColor = self._getTkColor(edgeColor, self.defaultLineColor)
        if edgeWidth is None:
            edgeWidth = self.defaultLineWidth
        new_item = self.create_rectangle(x1, y1, x2, y2, fill=fillColor, width=edgeWidth, outline=edgeColor)
        self._item_ids.append(new_item)

    def drawEllipse(self, x1, y1, x2, y2, edgeColor=None, edgeWidth=None, fillColor=None):
        fillColor = self._getTkColor(fillColor, self.defaultFillColor)
        edgeColor = self._getTkColor(edgeColor, self.defaultLineColor)
        if edgeWidth is None:
            edgeWidth = self.defaultLineWidth
        new_item = self.create_oval(x1, y1, x2, y2, fill=fillColor, outline=edgeColor, width=edgeWidth)
        self._item_ids.append(new_item)

    def drawArc(self, x1, y1, x2, y2, startAng=0, extent=360, edgeColor=None, edgeWidth=None, fillColor=None):
        fillColor = self._getTkColor(fillColor, self.defaultFillColor)
        edgeColor = self._getTkColor(edgeColor, self.defaultLineColor)
        if edgeWidth is None:
            edgeWidth = self.defaultLineWidth
        new_item = self.create_arc(x1, y1, x2, y2, start=startAng, extent=extent, fill=fillColor, width=edgeWidth, outline=edgeColor)
        self._item_ids.append(new_item)

    def drawPolygon(self, pointlist, edgeColor=None, edgeWidth=None, fillColor=None, closed=0):
        fillColor = self._getTkColor(fillColor, self.defaultFillColor)
        edgeColor = self._getTkColor(edgeColor, self.defaultLineColor)
        if edgeWidth is None:
            edgeWidth = self.defaultLineWidth
        if closed:
            new_item = self.create_polygon(pointlist, fill=fillColor, width=edgeWidth, outline=edgeColor)
        elif fillColor == self.__TRANSPARENT:
            d = {'fill': edgeColor, 'width': edgeWidth}
            new_item = self.create_line(pointlist, **d)
        else:
            new_item = self.create_polygon(pointlist, fill=fillColor, outline=self.__TRANSPARENT)
            self._item_ids.append(new_item)
            d = {'fill': edgeColor, 'width': edgeWidth}
            new_item = self.create_line(pointlist, **d)
        self._item_ids.append(new_item)

    def drawImage(self, image, x1, y1, x2=None, y2=None):
        try:
            from PIL import ImageTk
        except ImportError:
            raise NotImplementedError('drawImage - require the ImageTk module')
        w, h = image.size
        if not x2:
            x2 = w + x1
        if not y2:
            y2 = h + y1
        if w != x2 - x1 or h != y2 - y1:
            myimage = image.resize((x2 - x1, y2 - y1))
        else:
            myimage = image
        itk = ImageTk.PhotoImage(myimage, master=self)
        new_item = self.create_image(x1, y1, image=itk, anchor=Tkinter.NW)
        self._item_ids.append(new_item)
        self._images.append(itk)