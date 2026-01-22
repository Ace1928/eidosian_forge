import sys
def _bindArrows(widget, skipArrowKeys=False):
    widget.bind('<Down>', _tabRight)
    widget.bind('<Up>', _tabLeft)
    if not skipArrowKeys:
        widget.bind('<Right>', _tabRight)
        widget.bind('<Left>', _tabLeft)