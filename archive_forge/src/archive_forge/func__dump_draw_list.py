import ctypes
import weakref
import pyglet
from pyglet.gl import *
from pyglet.graphics import shader, vertexdomain
from pyglet.graphics.vertexarray import VertexArray
from pyglet.graphics.vertexbuffer import BufferObject
def _dump_draw_list(self):

    def dump(group, indent=''):
        print(indent, 'Begin group', group)
        domain_map = self.group_map[group]
        for _, domain in domain_map.items():
            print(indent, '  ', domain)
            for start, size in zip(*domain.allocator.get_allocated_regions()):
                print(indent, '    ', 'Region %d size %d:' % (start, size))
                for key, attribute in domain.attribute_names.items():
                    print(indent, '      ', end=' ')
                    try:
                        region = attribute.get_region(attribute.buffer, start, size)
                        print(key, region.array[:])
                    except:
                        print(key, '(unmappable)')
        for child in self.group_children.get(group, ()):
            dump(child, indent + '  ')
        print(indent, 'End group', group)
    print('Draw list for %r:' % self)
    for group in self.top_groups:
        dump(group)