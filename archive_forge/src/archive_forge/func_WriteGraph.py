import collections
import json
import sys
def WriteGraph(edges):
    """Print a graphviz graph to stdout.
  |edges| is a map of target to a list of other targets it depends on."""
    files = collections.defaultdict(list)
    for src, dst in edges.items():
        build_file, target_name, toolset = ParseTarget(src)
        files[build_file].append(src)
    print('digraph D {')
    print('  fontsize=8')
    print('  node [fontsize=8]')
    for filename, targets in files.items():
        if len(targets) == 1:
            target = targets[0]
            build_file, target_name, toolset = ParseTarget(target)
            print(f'  "{target}" [shape=box, label="{filename}\\n{target_name}"]')
        else:
            print('  subgraph "cluster_%s" {' % filename)
            print('    label = "%s"' % filename)
            for target in targets:
                build_file, target_name, toolset = ParseTarget(target)
                print(f'    "{target}" [label="{target_name}"]')
            print('  }')
    for src, dsts in edges.items():
        for dst in dsts:
            print(f'  "{src}" -> "{dst}"')
    print('}')