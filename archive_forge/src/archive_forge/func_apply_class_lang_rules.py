from kivy.event import EventDispatcher
from kivy.eventmanager import (
from kivy.factory import Factory
from kivy.properties import (
from kivy.graphics import (
from kivy.graphics.transformation import Matrix
from kivy.base import EventLoop
from kivy.lang import Builder
from kivy.context import get_current_context
from kivy.weakproxy import WeakProxy
from functools import partial
from itertools import islice
def apply_class_lang_rules(self, root=None, ignored_consts=set(), rule_children=None):
    """
        Method that is called by kivy to apply the kv rules of this widget's
        class.

        :Parameters:
            `root`: :class:`Widget`
                The root widget that instantiated this widget in kv, if the
                widget was instantiated in kv, otherwise ``None``.
            `ignored_consts`: set
                (internal) See :meth:`~kivy.lang.builder.BuilderBase.apply`.
            `rule_children`: list
                (internal) See :meth:`~kivy.lang.builder.BuilderBase.apply`.

        This is useful to be able to execute code before/after the class kv
        rules are applied to the widget. E.g. if the kv code requires some
        properties to be initialized before it is used in a binding rule.
        If overwriting remember to call ``super``, otherwise the kv rules will
        not be applied.

        In the following example,

        .. code-block:: python

            class MyWidget(Widget):
                pass

            class OtherWidget(MyWidget):
                pass

        .. code-block:: kv

        <MyWidget>:
            my_prop: some_value

        <OtherWidget>:
            other_prop: some_value

        When ``OtherWidget`` is instantiated with ``OtherWidget()``, the
        widget's :meth:`apply_class_lang_rules` is called and it applies the
        kv rules of this class - ``<MyWidget>`` and ``<OtherWidget>``.

        Similarly, when the widget is instantiated from kv, e.g.

        .. code-block:: kv

            <MyBox@BoxLayout>:
                height: 55
                OtherWidget:
                    width: 124

        ``OtherWidget``'s :meth:`apply_class_lang_rules` is called and it
        applies the kv rules of this class - ``<MyWidget>`` and
        ``<OtherWidget>``.

        .. note::

            It applies only the class rules not the instance rules. I.e. in the
            above kv example in the ``MyBox`` rule when ``OtherWidget`` is
            instantiated, its :meth:`apply_class_lang_rules` applies the
            ``<MyWidget>`` and ``<OtherWidget>`` rules to it - it does not
            apply the ``width: 124`` rule. The ``width: 124`` rule is part of
            the ``MyBox`` rule and is applied by the ``MyBox``'s instance's
            :meth:`apply_class_lang_rules`.

        .. versionchanged:: 1.11.0
        """
    Builder.apply(self, ignored_consts=ignored_consts, rule_children=rule_children)