from __future__ import absolute_import, division, print_function
def vlan_expander(data):
    expanded_list = []
    for each in data.split(','):
        if '-' in each:
            f, t = map(int, each.split('-'))
            expanded_list.extend(range(f, t + 1))
        else:
            expanded_list.append(int(each))
    return sorted(expanded_list)