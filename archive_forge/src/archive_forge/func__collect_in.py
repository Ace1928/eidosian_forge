from __future__ import absolute_import  #, unicode_literals
def _collect_in(self, target_list):
    for x in self.prepended_children:
        x._collect_in(target_list)
    stream_content = self.stream.getvalue()
    if stream_content:
        target_list.append(stream_content)