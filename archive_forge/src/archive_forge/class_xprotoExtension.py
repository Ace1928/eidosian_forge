import xcffib
import struct
import io
class xprotoExtension(xcffib.Extension):

    def CreateWindow(self, depth, wid, parent, x, y, width, height, border_width, _class, visual, value_mask, value_list, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIhhHHHHII', depth, wid, parent, x, y, width, height, border_width, _class, visual, value_mask))
        if value_mask & CW.BackPixmap:
            background_pixmap = value_list.pop(0)
            buf.write(struct.pack('=I', background_pixmap))
        if value_mask & CW.BackPixel:
            background_pixel = value_list.pop(0)
            buf.write(struct.pack('=I', background_pixel))
        if value_mask & CW.BorderPixmap:
            border_pixmap = value_list.pop(0)
            buf.write(struct.pack('=I', border_pixmap))
        if value_mask & CW.BorderPixel:
            border_pixel = value_list.pop(0)
            buf.write(struct.pack('=I', border_pixel))
        if value_mask & CW.BitGravity:
            bit_gravity = value_list.pop(0)
            buf.write(struct.pack('=I', bit_gravity))
        if value_mask & CW.WinGravity:
            win_gravity = value_list.pop(0)
            buf.write(struct.pack('=I', win_gravity))
        if value_mask & CW.BackingStore:
            backing_store = value_list.pop(0)
            buf.write(struct.pack('=I', backing_store))
        if value_mask & CW.BackingPlanes:
            backing_planes = value_list.pop(0)
            buf.write(struct.pack('=I', backing_planes))
        if value_mask & CW.BackingPixel:
            backing_pixel = value_list.pop(0)
            buf.write(struct.pack('=I', backing_pixel))
        if value_mask & CW.OverrideRedirect:
            override_redirect = value_list.pop(0)
            buf.write(struct.pack('=I', override_redirect))
        if value_mask & CW.SaveUnder:
            save_under = value_list.pop(0)
            buf.write(struct.pack('=I', save_under))
        if value_mask & CW.EventMask:
            event_mask = value_list.pop(0)
            buf.write(struct.pack('=I', event_mask))
        if value_mask & CW.DontPropagate:
            do_not_propogate_mask = value_list.pop(0)
            buf.write(struct.pack('=I', do_not_propogate_mask))
        if value_mask & CW.Colormap:
            colormap = value_list.pop(0)
            buf.write(struct.pack('=I', colormap))
        if value_mask & CW.Cursor:
            cursor = value_list.pop(0)
            buf.write(struct.pack('=I', cursor))
        return self.send_request(1, buf, is_checked=is_checked)

    def ChangeWindowAttributes(self, window, value_mask, value_list, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, value_mask))
        if value_mask & CW.BackPixmap:
            background_pixmap = value_list.pop(0)
            buf.write(struct.pack('=I', background_pixmap))
        if value_mask & CW.BackPixel:
            background_pixel = value_list.pop(0)
            buf.write(struct.pack('=I', background_pixel))
        if value_mask & CW.BorderPixmap:
            border_pixmap = value_list.pop(0)
            buf.write(struct.pack('=I', border_pixmap))
        if value_mask & CW.BorderPixel:
            border_pixel = value_list.pop(0)
            buf.write(struct.pack('=I', border_pixel))
        if value_mask & CW.BitGravity:
            bit_gravity = value_list.pop(0)
            buf.write(struct.pack('=I', bit_gravity))
        if value_mask & CW.WinGravity:
            win_gravity = value_list.pop(0)
            buf.write(struct.pack('=I', win_gravity))
        if value_mask & CW.BackingStore:
            backing_store = value_list.pop(0)
            buf.write(struct.pack('=I', backing_store))
        if value_mask & CW.BackingPlanes:
            backing_planes = value_list.pop(0)
            buf.write(struct.pack('=I', backing_planes))
        if value_mask & CW.BackingPixel:
            backing_pixel = value_list.pop(0)
            buf.write(struct.pack('=I', backing_pixel))
        if value_mask & CW.OverrideRedirect:
            override_redirect = value_list.pop(0)
            buf.write(struct.pack('=I', override_redirect))
        if value_mask & CW.SaveUnder:
            save_under = value_list.pop(0)
            buf.write(struct.pack('=I', save_under))
        if value_mask & CW.EventMask:
            event_mask = value_list.pop(0)
            buf.write(struct.pack('=I', event_mask))
        if value_mask & CW.DontPropagate:
            do_not_propogate_mask = value_list.pop(0)
            buf.write(struct.pack('=I', do_not_propogate_mask))
        if value_mask & CW.Colormap:
            colormap = value_list.pop(0)
            buf.write(struct.pack('=I', colormap))
        if value_mask & CW.Cursor:
            cursor = value_list.pop(0)
            buf.write(struct.pack('=I', cursor))
        return self.send_request(2, buf, is_checked=is_checked)

    def GetWindowAttributes(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(3, buf, GetWindowAttributesCookie, is_checked=is_checked)

    def DestroyWindow(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(4, buf, is_checked=is_checked)

    def DestroySubwindows(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(5, buf, is_checked=is_checked)

    def ChangeSaveSet(self, mode, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xI', mode, window))
        return self.send_request(6, buf, is_checked=is_checked)

    def ReparentWindow(self, window, parent, x, y, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIhh', window, parent, x, y))
        return self.send_request(7, buf, is_checked=is_checked)

    def MapWindow(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(8, buf, is_checked=is_checked)

    def MapSubwindows(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(9, buf, is_checked=is_checked)

    def UnmapWindow(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(10, buf, is_checked=is_checked)

    def UnmapSubwindows(self, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(11, buf, is_checked=is_checked)

    def ConfigureWindow(self, window, value_mask, value_list, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', window, value_mask))
        if value_mask & ConfigWindow.X:
            x = value_list.pop(0)
            buf.write(struct.pack('=i', x))
        if value_mask & ConfigWindow.Y:
            y = value_list.pop(0)
            buf.write(struct.pack('=i', y))
        if value_mask & ConfigWindow.Width:
            width = value_list.pop(0)
            buf.write(struct.pack('=I', width))
        if value_mask & ConfigWindow.Height:
            height = value_list.pop(0)
            buf.write(struct.pack('=I', height))
        if value_mask & ConfigWindow.BorderWidth:
            border_width = value_list.pop(0)
            buf.write(struct.pack('=I', border_width))
        if value_mask & ConfigWindow.Sibling:
            sibling = value_list.pop(0)
            buf.write(struct.pack('=I', sibling))
        if value_mask & ConfigWindow.StackMode:
            stack_mode = value_list.pop(0)
            buf.write(struct.pack('=I', stack_mode))
        return self.send_request(12, buf, is_checked=is_checked)

    def CirculateWindow(self, direction, window, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xI', direction, window))
        return self.send_request(13, buf, is_checked=is_checked)

    def GetGeometry(self, drawable, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', drawable))
        return self.send_request(14, buf, GetGeometryCookie, is_checked=is_checked)

    def QueryTree(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(15, buf, QueryTreeCookie, is_checked=is_checked)

    def InternAtom(self, only_if_exists, name_len, name, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xH2x', only_if_exists, name_len))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(16, buf, InternAtomCookie, is_checked=is_checked)

    def GetAtomName(self, atom, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', atom))
        return self.send_request(17, buf, GetAtomNameCookie, is_checked=is_checked)

    def ChangeProperty(self, mode, window, property, type, format, data_len, data, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIIB3xI', mode, window, property, type, format, data_len))
        buf.write(xcffib.pack_list(data, 'c'))
        return self.send_request(18, buf, is_checked=is_checked)

    def DeleteProperty(self, window, property, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', window, property))
        return self.send_request(19, buf, is_checked=is_checked)

    def GetProperty(self, delete, window, property, type, long_offset, long_length, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIIII', delete, window, property, type, long_offset, long_length))
        return self.send_request(20, buf, GetPropertyCookie, is_checked=is_checked)

    def ListProperties(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(21, buf, ListPropertiesCookie, is_checked=is_checked)

    def SetSelectionOwner(self, owner, selection, time, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', owner, selection, time))
        return self.send_request(22, buf, is_checked=is_checked)

    def GetSelectionOwner(self, selection, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', selection))
        return self.send_request(23, buf, GetSelectionOwnerCookie, is_checked=is_checked)

    def ConvertSelection(self, requestor, selection, target, property, time, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIII', requestor, selection, target, property, time))
        return self.send_request(24, buf, is_checked=is_checked)

    def SendEvent(self, propagate, destination, event_mask, event, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xII', propagate, destination, event_mask))
        buf.write(xcffib.pack_list(event, 'c'))
        return self.send_request(25, buf, is_checked=is_checked)

    def GrabPointer(self, owner_events, grab_window, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, time, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIHBBIII', owner_events, grab_window, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, time))
        return self.send_request(26, buf, GrabPointerCookie, is_checked=is_checked)

    def UngrabPointer(self, time, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', time))
        return self.send_request(27, buf, is_checked=is_checked)

    def GrabButton(self, owner_events, grab_window, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, button, modifiers, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIHBBIIBxH', owner_events, grab_window, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, button, modifiers))
        return self.send_request(28, buf, is_checked=is_checked)

    def UngrabButton(self, button, grab_window, modifiers, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIH2x', button, grab_window, modifiers))
        return self.send_request(29, buf, is_checked=is_checked)

    def ChangeActivePointerGrab(self, cursor, time, event_mask, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIH2x', cursor, time, event_mask))
        return self.send_request(30, buf, is_checked=is_checked)

    def GrabKeyboard(self, owner_events, grab_window, time, pointer_mode, keyboard_mode, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIBB2x', owner_events, grab_window, time, pointer_mode, keyboard_mode))
        return self.send_request(31, buf, GrabKeyboardCookie, is_checked=is_checked)

    def UngrabKeyboard(self, time, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', time))
        return self.send_request(32, buf, is_checked=is_checked)

    def GrabKey(self, owner_events, grab_window, modifiers, key, pointer_mode, keyboard_mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIHBBB3x', owner_events, grab_window, modifiers, key, pointer_mode, keyboard_mode))
        return self.send_request(33, buf, is_checked=is_checked)

    def UngrabKey(self, key, grab_window, modifiers, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIH2x', key, grab_window, modifiers))
        return self.send_request(34, buf, is_checked=is_checked)

    def AllowEvents(self, mode, time, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xI', mode, time))
        return self.send_request(35, buf, is_checked=is_checked)

    def GrabServer(self, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(36, buf, is_checked=is_checked)

    def UngrabServer(self, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(37, buf, is_checked=is_checked)

    def QueryPointer(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(38, buf, QueryPointerCookie, is_checked=is_checked)

    def GetMotionEvents(self, window, start, stop, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', window, start, stop))
        return self.send_request(39, buf, GetMotionEventsCookie, is_checked=is_checked)

    def TranslateCoordinates(self, src_window, dst_window, src_x, src_y, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIhh', src_window, dst_window, src_x, src_y))
        return self.send_request(40, buf, TranslateCoordinatesCookie, is_checked=is_checked)

    def WarpPointer(self, src_window, dst_window, src_x, src_y, src_width, src_height, dst_x, dst_y, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIhhHHhh', src_window, dst_window, src_x, src_y, src_width, src_height, dst_x, dst_y))
        return self.send_request(41, buf, is_checked=is_checked)

    def SetInputFocus(self, revert_to, focus, time, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xII', revert_to, focus, time))
        return self.send_request(42, buf, is_checked=is_checked)

    def GetInputFocus(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(43, buf, GetInputFocusCookie, is_checked=is_checked)

    def QueryKeymap(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(44, buf, QueryKeymapCookie, is_checked=is_checked)

    def OpenFont(self, fid, name_len, name, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', fid, name_len))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(45, buf, is_checked=is_checked)

    def CloseFont(self, font, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', font))
        return self.send_request(46, buf, is_checked=is_checked)

    def QueryFont(self, font, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', font))
        return self.send_request(47, buf, QueryFontCookie, is_checked=is_checked)

    def QueryTextExtents(self, font, string_len, string, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        buf.write(struct.pack('=B', string_len & 1))
        buf.write(struct.pack('=I', font))
        buf.write(xcffib.pack_list(string, CHAR2B))
        return self.send_request(48, buf, QueryTextExtentsCookie, is_checked=is_checked)

    def ListFonts(self, max_names, pattern_len, pattern, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', max_names, pattern_len))
        buf.write(xcffib.pack_list(pattern, 'c'))
        return self.send_request(49, buf, ListFontsCookie, is_checked=is_checked)

    def ListFontsWithInfo(self, max_names, pattern_len, pattern, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xHH', max_names, pattern_len))
        buf.write(xcffib.pack_list(pattern, 'c'))
        return self.send_request(50, buf, ListFontsWithInfoCookie, is_checked=is_checked)

    def SetFontPath(self, font_qty, font, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', font_qty))
        buf.write(xcffib.pack_list(font, STR))
        return self.send_request(51, buf, is_checked=is_checked)

    def GetFontPath(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(52, buf, GetFontPathCookie, is_checked=is_checked)

    def CreatePixmap(self, depth, pid, drawable, width, height, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIHH', depth, pid, drawable, width, height))
        return self.send_request(53, buf, is_checked=is_checked)

    def FreePixmap(self, pixmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', pixmap))
        return self.send_request(54, buf, is_checked=is_checked)

    def CreateGC(self, cid, drawable, value_mask, value_list, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', cid, drawable, value_mask))
        if value_mask & GC.Function:
            function = value_list.pop(0)
            buf.write(struct.pack('=I', function))
        if value_mask & GC.PlaneMask:
            plane_mask = value_list.pop(0)
            buf.write(struct.pack('=I', plane_mask))
        if value_mask & GC.Foreground:
            foreground = value_list.pop(0)
            buf.write(struct.pack('=I', foreground))
        if value_mask & GC.Background:
            background = value_list.pop(0)
            buf.write(struct.pack('=I', background))
        if value_mask & GC.LineWidth:
            line_width = value_list.pop(0)
            buf.write(struct.pack('=I', line_width))
        if value_mask & GC.LineStyle:
            line_style = value_list.pop(0)
            buf.write(struct.pack('=I', line_style))
        if value_mask & GC.CapStyle:
            cap_style = value_list.pop(0)
            buf.write(struct.pack('=I', cap_style))
        if value_mask & GC.JoinStyle:
            join_style = value_list.pop(0)
            buf.write(struct.pack('=I', join_style))
        if value_mask & GC.FillStyle:
            fill_style = value_list.pop(0)
            buf.write(struct.pack('=I', fill_style))
        if value_mask & GC.FillRule:
            fill_rule = value_list.pop(0)
            buf.write(struct.pack('=I', fill_rule))
        if value_mask & GC.Tile:
            tile = value_list.pop(0)
            buf.write(struct.pack('=I', tile))
        if value_mask & GC.Stipple:
            stipple = value_list.pop(0)
            buf.write(struct.pack('=I', stipple))
        if value_mask & GC.TileStippleOriginX:
            tile_stipple_x_origin = value_list.pop(0)
            buf.write(struct.pack('=i', tile_stipple_x_origin))
        if value_mask & GC.TileStippleOriginY:
            tile_stipple_y_origin = value_list.pop(0)
            buf.write(struct.pack('=i', tile_stipple_y_origin))
        if value_mask & GC.Font:
            font = value_list.pop(0)
            buf.write(struct.pack('=I', font))
        if value_mask & GC.SubwindowMode:
            subwindow_mode = value_list.pop(0)
            buf.write(struct.pack('=I', subwindow_mode))
        if value_mask & GC.GraphicsExposures:
            graphics_exposures = value_list.pop(0)
            buf.write(struct.pack('=I', graphics_exposures))
        if value_mask & GC.ClipOriginX:
            clip_x_origin = value_list.pop(0)
            buf.write(struct.pack('=i', clip_x_origin))
        if value_mask & GC.ClipOriginY:
            clip_y_origin = value_list.pop(0)
            buf.write(struct.pack('=i', clip_y_origin))
        if value_mask & GC.ClipMask:
            clip_mask = value_list.pop(0)
            buf.write(struct.pack('=I', clip_mask))
        if value_mask & GC.DashOffset:
            dash_offset = value_list.pop(0)
            buf.write(struct.pack('=I', dash_offset))
        if value_mask & GC.DashList:
            dashes = value_list.pop(0)
            buf.write(struct.pack('=I', dashes))
        if value_mask & GC.ArcMode:
            arc_mode = value_list.pop(0)
            buf.write(struct.pack('=I', arc_mode))
        return self.send_request(55, buf, is_checked=is_checked)

    def ChangeGC(self, gc, value_mask, value_list, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', gc, value_mask))
        if value_mask & GC.Function:
            function = value_list.pop(0)
            buf.write(struct.pack('=I', function))
        if value_mask & GC.PlaneMask:
            plane_mask = value_list.pop(0)
            buf.write(struct.pack('=I', plane_mask))
        if value_mask & GC.Foreground:
            foreground = value_list.pop(0)
            buf.write(struct.pack('=I', foreground))
        if value_mask & GC.Background:
            background = value_list.pop(0)
            buf.write(struct.pack('=I', background))
        if value_mask & GC.LineWidth:
            line_width = value_list.pop(0)
            buf.write(struct.pack('=I', line_width))
        if value_mask & GC.LineStyle:
            line_style = value_list.pop(0)
            buf.write(struct.pack('=I', line_style))
        if value_mask & GC.CapStyle:
            cap_style = value_list.pop(0)
            buf.write(struct.pack('=I', cap_style))
        if value_mask & GC.JoinStyle:
            join_style = value_list.pop(0)
            buf.write(struct.pack('=I', join_style))
        if value_mask & GC.FillStyle:
            fill_style = value_list.pop(0)
            buf.write(struct.pack('=I', fill_style))
        if value_mask & GC.FillRule:
            fill_rule = value_list.pop(0)
            buf.write(struct.pack('=I', fill_rule))
        if value_mask & GC.Tile:
            tile = value_list.pop(0)
            buf.write(struct.pack('=I', tile))
        if value_mask & GC.Stipple:
            stipple = value_list.pop(0)
            buf.write(struct.pack('=I', stipple))
        if value_mask & GC.TileStippleOriginX:
            tile_stipple_x_origin = value_list.pop(0)
            buf.write(struct.pack('=i', tile_stipple_x_origin))
        if value_mask & GC.TileStippleOriginY:
            tile_stipple_y_origin = value_list.pop(0)
            buf.write(struct.pack('=i', tile_stipple_y_origin))
        if value_mask & GC.Font:
            font = value_list.pop(0)
            buf.write(struct.pack('=I', font))
        if value_mask & GC.SubwindowMode:
            subwindow_mode = value_list.pop(0)
            buf.write(struct.pack('=I', subwindow_mode))
        if value_mask & GC.GraphicsExposures:
            graphics_exposures = value_list.pop(0)
            buf.write(struct.pack('=I', graphics_exposures))
        if value_mask & GC.ClipOriginX:
            clip_x_origin = value_list.pop(0)
            buf.write(struct.pack('=i', clip_x_origin))
        if value_mask & GC.ClipOriginY:
            clip_y_origin = value_list.pop(0)
            buf.write(struct.pack('=i', clip_y_origin))
        if value_mask & GC.ClipMask:
            clip_mask = value_list.pop(0)
            buf.write(struct.pack('=I', clip_mask))
        if value_mask & GC.DashOffset:
            dash_offset = value_list.pop(0)
            buf.write(struct.pack('=I', dash_offset))
        if value_mask & GC.DashList:
            dashes = value_list.pop(0)
            buf.write(struct.pack('=I', dashes))
        if value_mask & GC.ArcMode:
            arc_mode = value_list.pop(0)
            buf.write(struct.pack('=I', arc_mode))
        return self.send_request(56, buf, is_checked=is_checked)

    def CopyGC(self, src_gc, dst_gc, value_mask, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIII', src_gc, dst_gc, value_mask))
        return self.send_request(57, buf, is_checked=is_checked)

    def SetDashes(self, gc, dash_offset, dashes_len, dashes, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHH', gc, dash_offset, dashes_len))
        buf.write(xcffib.pack_list(dashes, 'B'))
        return self.send_request(58, buf, is_checked=is_checked)

    def SetClipRectangles(self, ordering, gc, clip_x_origin, clip_y_origin, rectangles_len, rectangles, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIhh', ordering, gc, clip_x_origin, clip_y_origin))
        buf.write(xcffib.pack_list(rectangles, RECTANGLE))
        return self.send_request(59, buf, is_checked=is_checked)

    def FreeGC(self, gc, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', gc))
        return self.send_request(60, buf, is_checked=is_checked)

    def ClearArea(self, exposures, window, x, y, width, height, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIhhHH', exposures, window, x, y, width, height))
        return self.send_request(61, buf, is_checked=is_checked)

    def CopyArea(self, src_drawable, dst_drawable, gc, src_x, src_y, dst_x, dst_y, width, height, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIhhhhHH', src_drawable, dst_drawable, gc, src_x, src_y, dst_x, dst_y, width, height))
        return self.send_request(62, buf, is_checked=is_checked)

    def CopyPlane(self, src_drawable, dst_drawable, gc, src_x, src_y, dst_x, dst_y, width, height, bit_plane, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIhhhhHHI', src_drawable, dst_drawable, gc, src_x, src_y, dst_x, dst_y, width, height, bit_plane))
        return self.send_request(63, buf, is_checked=is_checked)

    def PolyPoint(self, coordinate_mode, drawable, gc, points_len, points, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xII', coordinate_mode, drawable, gc))
        buf.write(xcffib.pack_list(points, POINT))
        return self.send_request(64, buf, is_checked=is_checked)

    def PolyLine(self, coordinate_mode, drawable, gc, points_len, points, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xII', coordinate_mode, drawable, gc))
        buf.write(xcffib.pack_list(points, POINT))
        return self.send_request(65, buf, is_checked=is_checked)

    def PolySegment(self, drawable, gc, segments_len, segments, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, gc))
        buf.write(xcffib.pack_list(segments, SEGMENT))
        return self.send_request(66, buf, is_checked=is_checked)

    def PolyRectangle(self, drawable, gc, rectangles_len, rectangles, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, gc))
        buf.write(xcffib.pack_list(rectangles, RECTANGLE))
        return self.send_request(67, buf, is_checked=is_checked)

    def PolyArc(self, drawable, gc, arcs_len, arcs, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, gc))
        buf.write(xcffib.pack_list(arcs, ARC))
        return self.send_request(68, buf, is_checked=is_checked)

    def FillPoly(self, drawable, gc, shape, coordinate_mode, points_len, points, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIBB2x', drawable, gc, shape, coordinate_mode))
        buf.write(xcffib.pack_list(points, POINT))
        return self.send_request(69, buf, is_checked=is_checked)

    def PolyFillRectangle(self, drawable, gc, rectangles_len, rectangles, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, gc))
        buf.write(xcffib.pack_list(rectangles, RECTANGLE))
        return self.send_request(70, buf, is_checked=is_checked)

    def PolyFillArc(self, drawable, gc, arcs_len, arcs, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', drawable, gc))
        buf.write(xcffib.pack_list(arcs, ARC))
        return self.send_request(71, buf, is_checked=is_checked)

    def PutImage(self, format, drawable, gc, width, height, dst_x, dst_y, left_pad, depth, data_len, data, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIHHhhBB2x', format, drawable, gc, width, height, dst_x, dst_y, left_pad, depth))
        buf.write(xcffib.pack_list(data, 'B'))
        return self.send_request(72, buf, is_checked=is_checked)

    def GetImage(self, format, drawable, x, y, width, height, plane_mask, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIhhHHI', format, drawable, x, y, width, height, plane_mask))
        return self.send_request(73, buf, GetImageCookie, is_checked=is_checked)

    def PolyText8(self, drawable, gc, x, y, items_len, items, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIhh', drawable, gc, x, y))
        buf.write(xcffib.pack_list(items, 'B'))
        return self.send_request(74, buf, is_checked=is_checked)

    def PolyText16(self, drawable, gc, x, y, items_len, items, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIhh', drawable, gc, x, y))
        buf.write(xcffib.pack_list(items, 'B'))
        return self.send_request(75, buf, is_checked=is_checked)

    def ImageText8(self, string_len, drawable, gc, x, y, string, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIhh', string_len, drawable, gc, x, y))
        buf.write(xcffib.pack_list(string, 'c'))
        return self.send_request(76, buf, is_checked=is_checked)

    def ImageText16(self, string_len, drawable, gc, x, y, string, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIhh', string_len, drawable, gc, x, y))
        buf.write(xcffib.pack_list(string, CHAR2B))
        return self.send_request(77, buf, is_checked=is_checked)

    def CreateColormap(self, alloc, mid, window, visual, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIII', alloc, mid, window, visual))
        return self.send_request(78, buf, is_checked=is_checked)

    def FreeColormap(self, cmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', cmap))
        return self.send_request(79, buf, is_checked=is_checked)

    def CopyColormapAndFree(self, mid, src_cmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', mid, src_cmap))
        return self.send_request(80, buf, is_checked=is_checked)

    def InstallColormap(self, cmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', cmap))
        return self.send_request(81, buf, is_checked=is_checked)

    def UninstallColormap(self, cmap, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', cmap))
        return self.send_request(82, buf, is_checked=is_checked)

    def ListInstalledColormaps(self, window, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', window))
        return self.send_request(83, buf, ListInstalledColormapsCookie, is_checked=is_checked)

    def AllocColor(self, cmap, red, green, blue, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHHH2x', cmap, red, green, blue))
        return self.send_request(84, buf, AllocColorCookie, is_checked=is_checked)

    def AllocNamedColor(self, cmap, name_len, name, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', cmap, name_len))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(85, buf, AllocNamedColorCookie, is_checked=is_checked)

    def AllocColorCells(self, contiguous, cmap, colors, planes, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIHH', contiguous, cmap, colors, planes))
        return self.send_request(86, buf, AllocColorCellsCookie, is_checked=is_checked)

    def AllocColorPlanes(self, contiguous, cmap, colors, reds, greens, blues, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIHHHH', contiguous, cmap, colors, reds, greens, blues))
        return self.send_request(87, buf, AllocColorPlanesCookie, is_checked=is_checked)

    def FreeColors(self, cmap, plane_mask, pixels_len, pixels, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xII', cmap, plane_mask))
        buf.write(xcffib.pack_list(pixels, 'I'))
        return self.send_request(88, buf, is_checked=is_checked)

    def StoreColors(self, cmap, items_len, items, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', cmap))
        buf.write(xcffib.pack_list(items, COLORITEM))
        return self.send_request(89, buf, is_checked=is_checked)

    def StoreNamedColor(self, flags, cmap, pixel, name_len, name, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIIH2x', flags, cmap, pixel, name_len))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(90, buf, is_checked=is_checked)

    def QueryColors(self, cmap, pixels_len, pixels, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', cmap))
        buf.write(xcffib.pack_list(pixels, 'I'))
        return self.send_request(91, buf, QueryColorsCookie, is_checked=is_checked)

    def LookupColor(self, cmap, name_len, name, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIH2x', cmap, name_len))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(92, buf, LookupColorCookie, is_checked=is_checked)

    def CreateCursor(self, cid, source, mask, fore_red, fore_green, fore_blue, back_red, back_green, back_blue, x, y, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIHHHHHHHH', cid, source, mask, fore_red, fore_green, fore_blue, back_red, back_green, back_blue, x, y))
        return self.send_request(93, buf, is_checked=is_checked)

    def CreateGlyphCursor(self, cid, source_font, mask_font, source_char, mask_char, fore_red, fore_green, fore_blue, back_red, back_green, back_blue, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIIHHHHHHHH', cid, source_font, mask_font, source_char, mask_char, fore_red, fore_green, fore_blue, back_red, back_green, back_blue))
        return self.send_request(94, buf, is_checked=is_checked)

    def FreeCursor(self, cursor, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', cursor))
        return self.send_request(95, buf, is_checked=is_checked)

    def RecolorCursor(self, cursor, fore_red, fore_green, fore_blue, back_red, back_green, back_blue, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHHHHHH', cursor, fore_red, fore_green, fore_blue, back_red, back_green, back_blue))
        return self.send_request(96, buf, is_checked=is_checked)

    def QueryBestSize(self, _class, drawable, width, height, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xIHH', _class, drawable, width, height))
        return self.send_request(97, buf, QueryBestSizeCookie, is_checked=is_checked)

    def QueryExtension(self, name_len, name, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xH2x', name_len))
        buf.write(xcffib.pack_list(name, 'c'))
        return self.send_request(98, buf, QueryExtensionCookie, is_checked=is_checked)

    def ListExtensions(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(99, buf, ListExtensionsCookie, is_checked=is_checked)

    def ChangeKeyboardMapping(self, keycode_count, first_keycode, keysyms_per_keycode, keysyms, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xBB2x', keycode_count, first_keycode, keysyms_per_keycode))
        buf.write(xcffib.pack_list(keysyms, 'I'))
        return self.send_request(100, buf, is_checked=is_checked)

    def GetKeyboardMapping(self, first_keycode, count, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB', first_keycode, count))
        return self.send_request(101, buf, GetKeyboardMappingCookie, is_checked=is_checked)

    def ChangeKeyboardControl(self, value_mask, value_list, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', value_mask))
        if value_mask & KB.KeyClickPercent:
            key_click_percent = value_list.pop(0)
            buf.write(struct.pack('=i', key_click_percent))
        if value_mask & KB.BellPercent:
            bell_percent = value_list.pop(0)
            buf.write(struct.pack('=i', bell_percent))
        if value_mask & KB.BellPitch:
            bell_pitch = value_list.pop(0)
            buf.write(struct.pack('=i', bell_pitch))
        if value_mask & KB.BellDuration:
            bell_duration = value_list.pop(0)
            buf.write(struct.pack('=i', bell_duration))
        if value_mask & KB.Led:
            led = value_list.pop(0)
            buf.write(struct.pack('=I', led))
        if value_mask & KB.LedMode:
            led_mode = value_list.pop(0)
            buf.write(struct.pack('=I', led_mode))
        if value_mask & KB.Key:
            key = value_list.pop(0)
            buf.write(struct.pack('=I', key))
        if value_mask & KB.AutoRepeatMode:
            auto_repeat_mode = value_list.pop(0)
            buf.write(struct.pack('=I', auto_repeat_mode))
        return self.send_request(102, buf, is_checked=is_checked)

    def GetKeyboardControl(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(103, buf, GetKeyboardControlCookie, is_checked=is_checked)

    def Bell(self, percent, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xb2x', percent))
        return self.send_request(104, buf, is_checked=is_checked)

    def ChangePointerControl(self, acceleration_numerator, acceleration_denominator, threshold, do_acceleration, do_threshold, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xhhhBB', acceleration_numerator, acceleration_denominator, threshold, do_acceleration, do_threshold))
        return self.send_request(105, buf, is_checked=is_checked)

    def GetPointerControl(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(106, buf, GetPointerControlCookie, is_checked=is_checked)

    def SetScreenSaver(self, timeout, interval, prefer_blanking, allow_exposures, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xhhBB', timeout, interval, prefer_blanking, allow_exposures))
        return self.send_request(107, buf, is_checked=is_checked)

    def GetScreenSaver(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(108, buf, GetScreenSaverCookie, is_checked=is_checked)

    def ChangeHosts(self, mode, family, address_len, address, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2xBxH', mode, family, address_len))
        buf.write(xcffib.pack_list(address, 'B'))
        return self.send_request(109, buf, is_checked=is_checked)

    def ListHosts(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(110, buf, ListHostsCookie, is_checked=is_checked)

    def SetAccessControl(self, mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2x', mode))
        return self.send_request(111, buf, is_checked=is_checked)

    def SetCloseDownMode(self, mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2x', mode))
        return self.send_request(112, buf, is_checked=is_checked)

    def KillClient(self, resource, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', resource))
        return self.send_request(113, buf, is_checked=is_checked)

    def RotateProperties(self, window, atoms_len, delta, atoms, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIHh', window, atoms_len, delta))
        buf.write(xcffib.pack_list(atoms, 'I'))
        return self.send_request(114, buf, is_checked=is_checked)

    def ForceScreenSaver(self, mode, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2x', mode))
        return self.send_request(115, buf, is_checked=is_checked)

    def SetPointerMapping(self, map_len, map, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2x', map_len))
        buf.write(xcffib.pack_list(map, 'B'))
        return self.send_request(116, buf, SetPointerMappingCookie, is_checked=is_checked)

    def GetPointerMapping(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(117, buf, GetPointerMappingCookie, is_checked=is_checked)

    def SetModifierMapping(self, keycodes_per_modifier, keycodes, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xB2x', keycodes_per_modifier))
        buf.write(xcffib.pack_list(keycodes, 'B'))
        return self.send_request(118, buf, SetModifierMappingCookie, is_checked=is_checked)

    def GetModifierMapping(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(119, buf, GetModifierMappingCookie, is_checked=is_checked)

    def NoOperation(self, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(127, buf, is_checked=is_checked)