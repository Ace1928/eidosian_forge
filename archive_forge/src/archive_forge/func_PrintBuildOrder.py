import os
import re
import sys
import pretty_vcproj
def PrintBuildOrder(projects, deps):
    print('---------------------------------------')
    print('Build order                            ')
    print('---------------------------------------')
    print('--                                   --')
    built = []
    for project, _ in sorted(deps.items()):
        if project not in built:
            BuildProject(project, built, projects, deps)
    print('--                                   --')