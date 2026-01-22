from __future__ import annotations
from typing import TYPE_CHECKING, List
from ..iapi import strip_draw_info, strip_label_details
class Strips(List[strip]):
    """
    List of strips for a plot
    """
    facet: facet

    @staticmethod
    def from_facet(facet: facet) -> Strips:
        new = Strips()
        new.facet = facet
        new.setup()
        return new

    @property
    def axs(self) -> list[Axes]:
        return self.facet.axs

    @property
    def layout(self) -> Layout:
        return self.facet.layout

    @property
    def theme(self) -> theme:
        return self.facet.theme

    @property
    def top_strips(self) -> Strips:
        return Strips([s for s in self if s.position == 'top'])

    @property
    def right_strips(self) -> Strips:
        return Strips([s for s in self if s.position == 'right'])

    def draw(self):
        for s in self:
            s.draw()

    def setup(self) -> Self:
        """
        Calculate the box information for all strips

        It is stored in self.strip_info
        """
        for layout_info in self.layout.get_details():
            ax = self.axs[layout_info.panel_index]
            lst = self.facet.make_strips(layout_info, ax)
            self.extend(lst)
        return self