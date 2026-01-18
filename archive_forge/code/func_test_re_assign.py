import unittest
def test_re_assign(self):
    from kivy.uix.behaviors.knspace import knspace, KNSpaceBehavior
    from kivy.uix.widget import Widget

    class MyWidget(KNSpaceBehavior, Widget):
        pass
    w = MyWidget(knsname='construct_name2')
    self.assertEqual(knspace.construct_name2, w)
    w2 = MyWidget(knsname='construct_name2')
    self.assertEqual(knspace.construct_name2, w2)