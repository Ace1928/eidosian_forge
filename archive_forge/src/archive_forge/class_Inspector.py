import weakref
from functools import partial
from itertools import chain
from kivy.animation import Animation
from kivy.logger import Logger
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.weakproxy import WeakProxy
from kivy.properties import (
class Inspector(Factory.FloatLayout):
    widget = ObjectProperty(None, allownone=True)
    layout = ObjectProperty(None)
    widgettree = ObjectProperty(None)
    treeview = ObjectProperty(None)
    inspect_enabled = BooleanProperty(False)
    activated = BooleanProperty(False)
    widget_info = BooleanProperty(False)
    content = ObjectProperty(None)
    at_bottom = BooleanProperty(True)
    _update_widget_tree_ev = None

    def __init__(self, **kwargs):
        self.win = kwargs.pop('win', None)
        super(Inspector, self).__init__(**kwargs)
        self.avoid_bring_to_top = False
        with self.canvas.before:
            self.gcolor = Factory.Color(1, 0, 0, 0.25)
            Factory.PushMatrix()
            self.gtransform = Factory.Transform(Matrix())
            self.grect = Factory.Rectangle(size=(0, 0))
            Factory.PopMatrix()
        Clock.schedule_interval(self.update_widget_graphics, 0)

    def on_touch_down(self, touch):
        ret = super(Inspector, self).on_touch_down(touch)
        if ('button' not in touch.profile or touch.button == 'left') and (not ret) and self.inspect_enabled:
            self.highlight_at(*touch.pos)
            if touch.is_double_tap:
                self.inspect_enabled = False
                self.show_widget_info()
            ret = True
        return ret

    def on_touch_move(self, touch):
        ret = super(Inspector, self).on_touch_move(touch)
        if not ret and self.inspect_enabled:
            self.highlight_at(*touch.pos)
            ret = True
        return ret

    def on_touch_up(self, touch):
        ret = super(Inspector, self).on_touch_up(touch)
        if not ret and self.inspect_enabled:
            ret = True
        return ret

    def on_window_children(self, win, children):
        if self.avoid_bring_to_top or not self.activated:
            return
        self.avoid_bring_to_top = True
        win.remove_widget(self)
        win.add_widget(self)
        self.avoid_bring_to_top = False

    def highlight_at(self, x, y):
        widget = None
        win_children = self.win.children
        children = chain((c for c in win_children if isinstance(c, Factory.ModalView)), (c for c in reversed(win_children) if not isinstance(c, Factory.ModalView)))
        for child in children:
            if child is self:
                continue
            widget = self.pick(child, x, y)
            if widget:
                break
        self.highlight_widget(widget)

    def highlight_widget(self, widget, info=True, *largs):
        self.widget = widget
        if not widget:
            self.grect.size = (0, 0)
        if self.widget_info and info:
            self.show_widget_info()

    def update_widget_graphics(self, *largs):
        if not self.activated:
            return
        if self.widget is None:
            self.grect.size = (0, 0)
            return
        self.grect.size = self.widget.size
        matrix = self.widget.get_window_matrix()
        if self.gtransform.matrix.get() != matrix.get():
            self.gtransform.matrix = matrix

    def toggle_position(self, button):
        to_bottom = button.text == 'Move to Bottom'
        if to_bottom:
            button.text = 'Move to Top'
            if self.widget_info:
                Animation(top=250, t='out_quad', d=0.3).start(self.layout)
            else:
                Animation(top=60, t='out_quad', d=0.3).start(self.layout)
            bottom_bar = self.layout.children[1]
            self.layout.remove_widget(bottom_bar)
            self.layout.add_widget(bottom_bar)
        else:
            button.text = 'Move to Bottom'
            if self.widget_info:
                Animation(top=self.height, t='out_quad', d=0.3).start(self.layout)
            else:
                Animation(y=self.height - 60, t='out_quad', d=0.3).start(self.layout)
            bottom_bar = self.layout.children[1]
            self.layout.remove_widget(bottom_bar)
            self.layout.add_widget(bottom_bar)
        self.at_bottom = to_bottom

    def pick(self, widget, x, y):
        ret = None
        if hasattr(widget, 'visible') and (not widget.visible):
            return ret
        if widget.collide_point(x, y):
            ret = widget
            x2, y2 = widget.to_local(x, y)
            for child in reversed(widget.children):
                ret = self.pick(child, x2, y2) or ret
        return ret

    def on_activated(self, instance, activated):
        if not activated:
            self.grect.size = (0, 0)
            if self.at_bottom:
                anim = Animation(top=0, t='out_quad', d=0.3)
            else:
                anim = Animation(y=self.height, t='out_quad', d=0.3)
            anim.bind(on_complete=self.animation_close)
            anim.start(self.layout)
            self.widget = None
            self.widget_info = False
        else:
            self.win.add_widget(self)
            Logger.info('Inspector: inspector activated')
            if self.at_bottom:
                Animation(top=60, t='out_quad', d=0.3).start(self.layout)
            else:
                Animation(y=self.height - 60, t='out_quad', d=0.3).start(self.layout)
            ev = self._update_widget_tree_ev
            if ev is None:
                ev = self._update_widget_tree_ev = Clock.schedule_interval(self.update_widget_tree, 1)
            else:
                ev()
            self.update_widget_tree()

    def animation_close(self, instance, value):
        if not self.activated:
            self.inspect_enabled = False
            self.win.remove_widget(self)
            self.content.clear_widgets()
            treeview = self.treeview
            for node in list(treeview.iterate_all_nodes()):
                node.widget_ref = None
                treeview.remove_node(node)
            self._window_node = None
            if self._update_widget_tree_ev is not None:
                self._update_widget_tree_ev.cancel()
            widgettree = self.widgettree
            for node in list(widgettree.iterate_all_nodes()):
                widgettree.remove_node(node)
            Logger.info('Inspector: inspector deactivated')

    def show_widget_info(self):
        self.content.clear_widgets()
        widget = self.widget
        treeview = self.treeview
        for node in list(treeview.iterate_all_nodes())[:]:
            node.widget_ref = None
            treeview.remove_node(node)
        if not widget:
            if self.at_bottom:
                Animation(top=60, t='out_quad', d=0.3).start(self.layout)
            else:
                Animation(y=self.height - 60, t='out_quad', d=0.3).start(self.layout)
            self.widget_info = False
            return
        self.widget_info = True
        if self.at_bottom:
            Animation(top=250, t='out_quad', d=0.3).start(self.layout)
        else:
            Animation(top=self.height, t='out_quad', d=0.3).start(self.layout)
        for node in list(treeview.iterate_all_nodes())[:]:
            treeview.remove_node(node)
        keys = list(widget.properties().keys())
        keys.sort()
        node = None
        if type(widget) is WeakProxy:
            wk_widget = widget.__ref__
        else:
            wk_widget = weakref.ref(widget)
        for key in keys:
            node = TreeViewProperty(key=key, widget_ref=wk_widget)
            node.bind(is_selected=self.show_property)
            try:
                widget.bind(**{key: partial(self.update_node_content, weakref.ref(node))})
            except:
                pass
            treeview.add_node(node)

    def update_node_content(self, node, *largs):
        node = node()
        if node is None:
            return
        node.refresh = True
        node.refresh = False

    def keyboard_shortcut(self, win, scancode, *largs):
        modifiers = largs[-1]
        if scancode == 101 and set(modifiers) & {'ctrl'} and (not set(modifiers) & {'shift', 'alt', 'meta'}):
            self.activated = not self.activated
            if self.activated:
                self.inspect_enabled = True
            return True
        elif scancode == 27:
            if self.inspect_enabled:
                self.inspect_enabled = False
                return True
            if self.activated:
                self.activated = False
                return True

    def show_property(self, instance, value, key=None, index=-1, *largs):
        if value is False:
            return
        content = None
        if key is None:
            nested = False
            widget = instance.widget
            key = instance.key
            prop = widget.property(key)
            value = getattr(widget, key)
        else:
            nested = True
            widget = instance
            prop = None
        dtype = None
        if isinstance(prop, AliasProperty) or nested:
            if type(value) in (str, str):
                dtype = 'string'
            elif type(value) in (int, float):
                dtype = 'numeric'
            elif type(value) in (tuple, list):
                dtype = 'list'
        if isinstance(prop, NumericProperty) or dtype == 'numeric':
            content = Factory.TextInput(text=str(value) or '', multiline=False)
            content.bind(text=partial(self.save_property_numeric, widget, key, index))
        elif isinstance(prop, StringProperty) or dtype == 'string':
            content = Factory.TextInput(text=value or '', multiline=True)
            content.bind(text=partial(self.save_property_text, widget, key, index))
        elif isinstance(prop, ListProperty) or isinstance(prop, ReferenceListProperty) or isinstance(prop, VariableListProperty) or (dtype == 'list'):
            content = Factory.GridLayout(cols=1, size_hint_y=None)
            content.bind(minimum_height=content.setter('height'))
            for i, item in enumerate(value):
                button = Factory.Button(text=repr(item), size_hint_y=None, height=44)
                if isinstance(item, Factory.Widget):
                    button.bind(on_release=partial(self.highlight_widget, item, False))
                else:
                    button.bind(on_release=partial(self.show_property, widget, item, key, i))
                content.add_widget(button)
        elif isinstance(prop, OptionProperty):
            content = Factory.GridLayout(cols=1, size_hint_y=None)
            content.bind(minimum_height=content.setter('height'))
            for option in prop.options:
                button = Factory.ToggleButton(text=option, state='down' if option == value else 'normal', group=repr(content.uid), size_hint_y=None, height=44)
                button.bind(on_press=partial(self.save_property_option, widget, key))
                content.add_widget(button)
        elif isinstance(prop, ObjectProperty):
            if isinstance(value, Factory.Widget):
                content = Factory.Button(text=repr(value))
                content.bind(on_release=partial(self.highlight_widget, value))
            elif isinstance(value, Factory.Texture):
                content = Factory.Image(texture=value)
            else:
                content = Factory.Label(text=repr(value))
        elif isinstance(prop, BooleanProperty):
            state = 'down' if value else 'normal'
            content = Factory.ToggleButton(text=key, state=state)
            content.bind(on_release=partial(self.save_property_boolean, widget, key, index))
        self.content.clear_widgets()
        if content:
            self.content.add_widget(content)

    def save_property_numeric(self, widget, key, index, instance, value):
        try:
            if index >= 0:
                getattr(widget, key)[index] = float(instance.text)
            else:
                setattr(widget, key, float(instance.text))
        except:
            pass

    def save_property_text(self, widget, key, index, instance, value):
        try:
            if index >= 0:
                getattr(widget, key)[index] = instance.text
            else:
                setattr(widget, key, instance.text)
        except:
            pass

    def save_property_boolean(self, widget, key, index, instance):
        try:
            value = instance.state == 'down'
            if index >= 0:
                getattr(widget, key)[index] = value
            else:
                setattr(widget, key, value)
        except:
            pass

    def save_property_option(self, widget, key, instance, *largs):
        try:
            setattr(widget, key, instance.text)
        except:
            pass

    def _update_widget_tree_node(self, node, widget, is_open=False):
        tree = self.widgettree
        update_nodes = []
        nodes = {}
        for cnode in node.nodes[:]:
            try:
                nodes[cnode.widget] = cnode
            except ReferenceError:
                pass
            tree.remove_node(cnode)
        for child in widget.children:
            if child is self:
                continue
            if child in nodes:
                cnode = tree.add_node(nodes[child], node)
            else:
                cnode = tree.add_node(TreeViewWidget(text=child.__class__.__name__, widget=child.proxy_ref, is_open=is_open), node)
            update_nodes.append((cnode, child))
        return update_nodes

    def update_widget_tree(self, *args):
        if not hasattr(self, '_window_node') or not self._window_node:
            self._window_node = self.widgettree.add_node(TreeViewWidget(text='Window', widget=self.win, is_open=True))
        nodes = self._update_widget_tree_node(self._window_node, self.win, is_open=True)
        while nodes:
            ntmp = nodes[:]
            nodes = []
            for node in ntmp:
                nodes += self._update_widget_tree_node(*node)
        self.widgettree.update_selected_widget(self.widget)