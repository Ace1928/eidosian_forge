import pytest
class TestLayout_only_a_part_of_the_data_is_visible:

    def compute_layout(self, *, n_cols, n_rows, orientation, n_data, scroll_to, clock):
        """Returns {view-index: pos, view-index: pos, ...}"""
        from textwrap import dedent
        from kivy.lang import Builder
        rv = Builder.load_string(dedent(f"\n            RecycleView:\n                viewclass: 'Widget'\n                size: 100, 100\n                data: ({{}} for __ in range({n_data}))\n                RecycleGridLayout:\n                    id: layout\n                    cols: {n_cols}\n                    rows: {n_rows}\n                    orientation: '{orientation}'\n                    default_size_hint: None, None\n                    default_size: 100, 100\n                    size_hint: None, None\n                    size: self.minimum_size\n            "))
        clock.tick()
        layout = rv.ids.layout
        x, y = scroll_to
        scrollable_width = layout.width - rv.width
        if scrollable_width:
            rv.scroll_x = x / scrollable_width
        scrollable_height = layout.height - rv.height
        if scrollable_height:
            rv.scroll_y = y / scrollable_height
        clock.tick()
        return {layout.get_view_index_at(c.center): tuple(c.pos) for c in layout.children}

    @pytest.mark.parametrize('n_cols, n_rows', [(4, None), (None, 1), (4, 1)])
    @pytest.mark.parametrize('orientation', 'lr-tb lr-bt tb-lr bt-lr'.split())
    def test_4x1_lr(self, kivy_clock, orientation, n_cols, n_rows):
        assert {1: (100, 0), 2: (200, 0)} == self.compute_layout(n_data=4, orientation=orientation, n_cols=n_cols, n_rows=n_rows, scroll_to=(150, 0), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(4, None), (None, 1), (4, 1)])
    @pytest.mark.parametrize('orientation', 'rl-tb rl-bt tb-rl bt-rl'.split())
    def test_4x1_rl(self, kivy_clock, orientation, n_cols, n_rows):
        assert {1: (200, 0), 2: (100, 0)} == self.compute_layout(n_data=4, orientation=orientation, n_cols=n_cols, n_rows=n_rows, scroll_to=(150, 0), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 4), (1, 4)])
    @pytest.mark.parametrize('orientation', 'tb-lr tb-rl lr-tb rl-tb'.split())
    def test_1x4_tb(self, kivy_clock, orientation, n_cols, n_rows):
        assert {1: (0, 200), 2: (0, 100)} == self.compute_layout(n_data=4, orientation=orientation, n_cols=n_cols, n_rows=n_rows, scroll_to=(0, 150), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(1, None), (None, 4), (1, 4)])
    @pytest.mark.parametrize('orientation', 'bt-lr bt-rl lr-bt rl-bt'.split())
    def test_1x4_bt(self, kivy_clock, orientation, n_cols, n_rows):
        assert {1: (0, 100), 2: (0, 200)} == self.compute_layout(n_data=4, orientation=orientation, n_cols=n_cols, n_rows=n_rows, scroll_to=(0, 150), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
    def test_3x3_lr_tb(self, kivy_clock, n_cols, n_rows):
        assert {4: (100, 100), 5: (200, 100), 7: (100, 0)} == self.compute_layout(n_data=8, orientation='lr-tb', n_cols=n_cols, n_rows=n_rows, scroll_to=(150, 50), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
    def test_3x3_lr_bt(self, kivy_clock, n_cols, n_rows):
        assert {4: (100, 100), 5: (200, 100), 7: (100, 200)} == self.compute_layout(n_data=8, orientation='lr-bt', n_cols=n_cols, n_rows=n_rows, scroll_to=(150, 150), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
    def test_3x3_rl_tb(self, kivy_clock, n_cols, n_rows):
        assert {4: (100, 100), 5: (0, 100), 7: (100, 0)} == self.compute_layout(n_data=8, orientation='rl-tb', n_cols=n_cols, n_rows=n_rows, scroll_to=(50, 50), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
    def test_3x3_rl_bt(self, kivy_clock, n_cols, n_rows):
        assert {4: (100, 100), 5: (0, 100), 7: (100, 200)} == self.compute_layout(n_data=8, orientation='rl-bt', n_cols=n_cols, n_rows=n_rows, scroll_to=(50, 150), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
    def test_3x3_tb_lr(self, kivy_clock, n_cols, n_rows):
        assert {4: (100, 100), 5: (100, 0), 7: (200, 100)} == self.compute_layout(n_data=8, orientation='tb-lr', n_cols=n_cols, n_rows=n_rows, scroll_to=(150, 50), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
    def test_3x3_tb_rl(self, kivy_clock, n_cols, n_rows):
        assert {4: (100, 100), 5: (100, 0), 7: (0, 100)} == self.compute_layout(n_data=8, orientation='tb-rl', n_cols=n_cols, n_rows=n_rows, scroll_to=(50, 50), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
    def test_3x3_bt_lr(self, kivy_clock, n_cols, n_rows):
        assert {4: (100, 100), 5: (100, 200), 7: (200, 100)} == self.compute_layout(n_data=8, orientation='bt-lr', n_cols=n_cols, n_rows=n_rows, scroll_to=(150, 150), clock=kivy_clock)

    @pytest.mark.parametrize('n_cols, n_rows', [(3, None), (None, 3), (3, 3)])
    def test_3x3_bt_rl(self, kivy_clock, n_cols, n_rows):
        assert {4: (100, 100), 5: (100, 200), 7: (0, 100)} == self.compute_layout(n_data=8, orientation='bt-rl', n_cols=n_cols, n_rows=n_rows, scroll_to=(50, 150), clock=kivy_clock)