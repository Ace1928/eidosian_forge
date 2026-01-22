from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
class CompositeTestCase(ElementTestCase):
    """
    Test case for trees involving both + (Layout) and * (Overlay)
    """

    def test_composite1(self):
        t = self.el1 * self.el2 + self.el1 * self.el2
        self.assertEqual(t.keys(), [('Overlay', 'I'), ('Overlay', 'II')])

    def test_composite_relabelled_value1(self):
        t = self.el1 * self.el2 + (self.el1 * self.el2).relabel(group='Val2')
        self.assertEqual(t.keys(), [('Overlay', 'I'), ('Val2', 'I')])

    def test_composite_relabelled_label1(self):
        t = self.el1 * self.el2 + (self.el1 * self.el2).relabel(group='Val1', label='Label2')
        self.assertEqual(t.keys(), [('Overlay', 'I'), ('Val1', 'Label2')])

    def test_composite_relabelled_label2(self):
        t = (self.el1 * self.el2).relabel(label='Label1') + (self.el1 * self.el2).relabel(group='Val1', label='Label2')
        self.assertEqual(t.keys(), [('Overlay', 'Label1'), ('Val1', 'Label2')])

    def test_composite_relabelled_value2(self):
        t = (self.el1 * self.el2).relabel(group='Val1') + (self.el1 * self.el2).relabel(group='Val2')
        self.assertEqual(t.keys(), [('Val1', 'I'), ('Val2', 'I')])

    def test_composite_relabelled_value_and_label(self):
        t = (self.el1 * self.el2).relabel(group='Val1', label='Label1') + (self.el1 * self.el2).relabel(group='Val2', label='Label2')
        self.assertEqual(t.keys(), [('Val1', 'Label1'), ('Val2', 'Label2')])

    def test_triple_composite_relabelled_value_and_label_keys(self):
        t = self.el1 * self.el2 + (self.el1 * self.el2).relabel(group='Val1', label='Label1') + (self.el1 * self.el2).relabel(group='Val2', label='Label2')
        excepted_keys = [('Overlay', 'I'), ('Val1', 'Label1'), ('Val2', 'Label2')]
        self.assertEqual(t.keys(), excepted_keys)

    def test_deep_composite_values(self):
        o1 = self.el1 * self.el2
        o2 = self.el1 * self.el2
        o3 = self.el7 * self.el8
        t = o1 + o2 + o3
        self.assertEqual(t.values(), [o1, o2, o3])

    def test_deep_composite_keys(self):
        o1 = self.el1 * self.el2
        o2 = self.el1 * self.el2
        o3 = self.el7 * self.el8
        t = o1 + o2 + o3
        expected_keys = [('Overlay', 'I'), ('Overlay', 'II'), ('ValA', 'I')]
        self.assertEqual(t.keys(), expected_keys)

    def test_deep_composite_indexing(self):
        o1 = self.el1 * self.el2
        o2 = self.el1 * self.el2
        o3 = self.el7 * self.el8
        t = o1 + o2 + o3
        expected_keys = [('Overlay', 'I'), ('Overlay', 'II'), ('ValA', 'I')]
        self.assertEqual(t.keys(), expected_keys)
        self.assertEqual(t.ValA.I, o3)
        self.assertEqual(t.ValA.I.ValA.LabelA, self.el7)
        self.assertEqual(t.ValA.I.ValA.LabelB, self.el8)

    def test_deep_composite_getitem(self):
        o1 = self.el1 * self.el2
        o2 = self.el1 * self.el2
        o3 = self.el7 * self.el8
        t = o1 + o2 + o3
        expected_keys = [('Overlay', 'I'), ('Overlay', 'II'), ('ValA', 'I')]
        self.assertEqual(t.keys(), expected_keys)
        self.assertEqual(t['ValA']['I'], o3)
        self.assertEqual(t['ValA']['I'].get('ValA').get('LabelA'), self.el7)
        self.assertEqual(t['ValA']['I'].get('ValA').get('LabelB'), self.el8)

    def test_invalid_tree_structure(self):
        try:
            (self.el1 + self.el2) * (self.el1 + self.el2)
        except TypeError as e:
            self.assertEqual(str(e), "unsupported operand type(s) for *: 'Layout' and 'Layout'")