import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
class NavigationToolbar2QT(NavigationToolbar2, QtWidgets.QToolBar):
    _message = QtCore.Signal(str)
    message = _api.deprecate_privatize_attribute('3.8')
    toolitems = [*NavigationToolbar2.toolitems]
    toolitems.insert([name for name, *_ in toolitems].index('Subplots') + 1, ('Customize', 'Edit axis, curve and image parameters', 'qt4_editor_options', 'edit_parameters'))

    def __init__(self, canvas, parent=None, coordinates=True):
        """coordinates: should we show the coordinates on the right?"""
        QtWidgets.QToolBar.__init__(self, parent)
        self.setAllowedAreas(QtCore.Qt.ToolBarArea(_to_int(QtCore.Qt.ToolBarArea.TopToolBarArea) | _to_int(QtCore.Qt.ToolBarArea.BottomToolBarArea)))
        self.coordinates = coordinates
        self._actions = {}
        self._subplot_dialog = None
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                a = self.addAction(self._icon(image_file + '.png'), text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['zoom', 'pan']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)
        if self.coordinates:
            self.locLabel = QtWidgets.QLabel('', self)
            self.locLabel.setAlignment(QtCore.Qt.AlignmentFlag(_to_int(QtCore.Qt.AlignmentFlag.AlignRight) | _to_int(QtCore.Qt.AlignmentFlag.AlignVCenter)))
            self.locLabel.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)
        NavigationToolbar2.__init__(self, canvas)

    def _icon(self, name):
        """
        Construct a `.QIcon` from an image file *name*, including the extension
        and relative to Matplotlib's "images" data directory.
        """
        path_regular = cbook._get_data_path('images', name)
        path_large = path_regular.with_name(path_regular.name.replace('.png', '_large.png'))
        filename = str(path_large if path_large.exists() else path_regular)
        pm = QtGui.QPixmap(filename)
        pm.setDevicePixelRatio(self.devicePixelRatioF() or 1)
        if self.palette().color(self.backgroundRole()).value() < 128:
            icon_color = self.palette().color(self.foregroundRole())
            mask = pm.createMaskFromColor(QtGui.QColor('black'), QtCore.Qt.MaskMode.MaskOutColor)
            pm.fill(icon_color)
            pm.setMask(mask)
        return QtGui.QIcon(pm)

    def edit_parameters(self):
        axes = self.canvas.figure.get_axes()
        if not axes:
            QtWidgets.QMessageBox.warning(self.canvas.parent(), 'Error', 'There are no axes to edit.')
            return
        elif len(axes) == 1:
            ax, = axes
        else:
            titles = [ax.get_label() or ax.get_title() or ax.get_title('left') or ax.get_title('right') or ' - '.join(filter(None, [ax.get_xlabel(), ax.get_ylabel()])) or f'<anonymous {type(ax).__name__}>' for ax in axes]
            duplicate_titles = [title for title in titles if titles.count(title) > 1]
            for i, ax in enumerate(axes):
                if titles[i] in duplicate_titles:
                    titles[i] += f' (id: {id(ax):#x})'
            item, ok = QtWidgets.QInputDialog.getItem(self.canvas.parent(), 'Customize', 'Select axes:', titles, 0, False)
            if not ok:
                return
            ax = axes[titles.index(item)]
        figureoptions.figure_edit(ax, self)

    def _update_buttons_checked(self):
        if 'pan' in self._actions:
            self._actions['pan'].setChecked(self.mode.name == 'PAN')
        if 'zoom' in self._actions:
            self._actions['zoom'].setChecked(self.mode.name == 'ZOOM')

    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    def set_message(self, s):
        self._message.emit(s)
        if self.coordinates:
            self.locLabel.setText(s)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self):
        self.canvas.drawRectangle(None)

    def configure_subplots(self):
        if self._subplot_dialog is None:
            self._subplot_dialog = SubplotToolQt(self.canvas.figure, self.canvas.parent())
            self.canvas.mpl_connect('close_event', lambda e: self._subplot_dialog.reject())
        self._subplot_dialog.update_from_current_subplotpars()
        self._subplot_dialog.show()
        return self._subplot_dialog

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()
        startpath = os.path.expanduser(mpl.rcParams['savefig.directory'])
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = ' '.join(['*.%s' % ext for ext in exts])
            filter = f'{name} ({exts_list})'
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)
        fname, filter = QtWidgets.QFileDialog.getSaveFileName(self.canvas.parent(), 'Choose a filename to save to', start, filters, selectedFilter)
        if fname:
            if startpath != '':
                mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
            try:
                self.canvas.figure.savefig(fname)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Error saving file', str(e), QtWidgets.QMessageBox.StandardButton.Ok, QtWidgets.QMessageBox.StandardButton.NoButton)

    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
        if 'back' in self._actions:
            self._actions['back'].setEnabled(can_backward)
        if 'forward' in self._actions:
            self._actions['forward'].setEnabled(can_forward)