import os
import re
import sys
import pretty_vcproj
def PrintDependencies(projects, deps):
    print('---------------------------------------')
    print('Dependencies for all projects')
    print('---------------------------------------')
    print('--                                   --')
    for project, dep_list in sorted(deps.items()):
        print('Project : %s' % project)
        print('Path : %s' % projects[project][0])
        if dep_list:
            for dep in dep_list:
                print('  - %s' % dep)
        print('')
    print('--                                   --')