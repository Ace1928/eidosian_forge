import ctypes
import weakref
import pyglet
from pyglet.gl import *
from pyglet.graphics import shader, vertexdomain
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import BufferObject
def _update_draw_list(self):
    """Visit group tree in preorder and create a list of bound methods
        to call.
        """

    def visit(group):
        draw_list = []
        domain_map = self.group_map[group]
        for (formats, mode, indexed, program_id), domain in list(domain_map.items()):
            if domain.is_empty:
                del domain_map[formats, mode, indexed, program_id]
                continue
            draw_list.append((lambda d, m: lambda: d.draw(m))(domain, mode))
        children = self.group_children.get(group)
        if children:
            children.sort()
            for child in list(children):
                if child.visible:
                    draw_list.extend(visit(child))
        if children or domain_map:
            return [group.set_state] + draw_list + [group.unset_state]
        else:
            del self.group_map[group]
            group._assigned_batches.remove(self)
            if group.parent:
                self.group_children[group.parent].remove(group)
            try:
                del self.group_children[group]
            except KeyError:
                pass
            try:
                self.top_groups.remove(group)
            except ValueError:
                pass
            return []
    self._draw_list = []
    self.top_groups.sort()
    for group in list(self.top_groups):
        if group.visible:
            self._draw_list.extend(visit(group))
    self._draw_list_dirty = False
    if _debug_graphics_batch:
        self._dump_draw_list()