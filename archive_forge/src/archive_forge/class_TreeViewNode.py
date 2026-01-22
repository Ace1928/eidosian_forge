from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
class TreeViewNode(object):
    """TreeViewNode class, used to build a node class for a TreeView object.
    """

    def __init__(self, **kwargs):
        if self.__class__ is TreeViewNode:
            raise TreeViewException('You cannot use directly TreeViewNode.')
        super(TreeViewNode, self).__init__(**kwargs)
    is_leaf = BooleanProperty(True)
    'Boolean to indicate whether this node is a leaf or not. Used to adjust\n    the graphical representation.\n\n    :attr:`is_leaf` is a :class:`~kivy.properties.BooleanProperty` and defaults\n    to True. It is automatically set to False when child is added.\n    '
    is_open = BooleanProperty(False)
    'Boolean to indicate whether this node is opened or not, in case there\n    are child nodes. This is used to adjust the graphical representation.\n\n    .. warning::\n\n        This property is automatically set by the :class:`TreeView`. You can\n        read but not write it.\n\n    :attr:`is_open` is a :class:`~kivy.properties.BooleanProperty` and defaults\n    to False.\n    '
    is_loaded = BooleanProperty(False)
    'Boolean to indicate whether this node is already loaded or not. This\n    property is used only if the :class:`TreeView` uses asynchronous loading.\n\n    :attr:`is_loaded` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    is_selected = BooleanProperty(False)
    'Boolean to indicate whether this node is selected or not. This is used\n    adjust the graphical representation.\n\n    .. warning::\n\n        This property is automatically set by the :class:`TreeView`. You can\n        read but not write it.\n\n    :attr:`is_selected` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    no_selection = BooleanProperty(False)
    'Boolean used to indicate whether selection of the node is allowed or\n     not.\n\n    :attr:`no_selection` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to False.\n    '
    nodes = ListProperty([])
    'List of nodes. The nodes list is different than the children list. A\n    node in the nodes list represents a node on the tree. An item in the\n    children list represents the widget associated with the node.\n\n    .. warning::\n\n        This property is automatically set by the :class:`TreeView`. You can\n        read but not write it.\n\n    :attr:`nodes` is a :class:`~kivy.properties.ListProperty` and defaults to\n    [].\n    '
    parent_node = ObjectProperty(None, allownone=True)
    'Parent node. This attribute is needed because the :attr:`parent` can be\n    None when the node is not displayed.\n\n    .. versionadded:: 1.0.7\n\n    :attr:`parent_node` is an :class:`~kivy.properties.ObjectProperty` and\n    defaults to None.\n    '
    level = NumericProperty(-1)
    'Level of the node.\n\n    :attr:`level` is a :class:`~kivy.properties.NumericProperty` and defaults\n    to -1.\n    '
    color_selected = ColorProperty([0.3, 0.3, 0.3, 1.0])
    'Background color of the node when the node is selected.\n\n    :attr:`color_selected` is a :class:`~kivy.properties.ColorProperty` and\n    defaults to [.1, .1, .1, 1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    odd = BooleanProperty(False)
    '\n    This property is set by the TreeView widget automatically and is read-only.\n\n    :attr:`odd` is a :class:`~kivy.properties.BooleanProperty` and defaults to\n    False.\n    '
    odd_color = ColorProperty([1.0, 1.0, 1.0, 0.0])
    'Background color of odd nodes when the node is not selected.\n\n    :attr:`odd_color` is a :class:`~kivy.properties.ColorProperty` and defaults\n    to [1., 1., 1., 0.].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '
    even_color = ColorProperty([0.5, 0.5, 0.5, 0.1])
    'Background color of even nodes when the node is not selected.\n\n    :attr:`bg_color` is a :class:`~kivy.properties.ColorProperty` and defaults\n    to [.5, .5, .5, .1].\n\n    .. versionchanged:: 2.0.0\n        Changed from :class:`~kivy.properties.ListProperty` to\n        :class:`~kivy.properties.ColorProperty`.\n    '