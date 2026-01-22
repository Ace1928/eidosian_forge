import datetime
from redis.utils import str_if_bytes
def _parse_slots(slot_ranges):
    slots, migrations = ([], [])
    for s_range in slot_ranges:
        if '->-' in s_range:
            slot_id, dst_node_id = s_range[1:-1].split('->-', 1)
            migrations.append({'slot': slot_id, 'node_id': dst_node_id, 'state': 'migrating'})
        elif '-<-' in s_range:
            slot_id, src_node_id = s_range[1:-1].split('-<-', 1)
            migrations.append({'slot': slot_id, 'node_id': src_node_id, 'state': 'importing'})
        else:
            s_range = [sl for sl in s_range.split('-')]
            slots.append(s_range)
    return (slots, migrations)