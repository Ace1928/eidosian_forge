import os
import re
import sys
import pretty_vcproj
def ParseSolution(solution_file):
    projects = dict()
    dependencies = dict()
    begin_project = re.compile('^Project\\("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}"\\) = "(.*)", "(.*)", "(.*)"$')
    end_project = re.compile('^EndProject$')
    begin_dep = re.compile('ProjectSection\\(ProjectDependencies\\) = postProject$')
    end_dep = re.compile('EndProjectSection$')
    dep_line = re.compile(' *({.*}) = ({.*})$')
    in_deps = False
    solution = open(solution_file)
    for line in solution:
        results = begin_project.search(line)
        if results:
            if results.group(1).find('icu') != -1:
                continue
            current_project = results.group(1).replace('_gyp', '')
            projects[current_project] = [results.group(2).replace('_gyp', ''), results.group(3), results.group(2)]
            dependencies[current_project] = []
            continue
        results = end_project.search(line)
        if results:
            current_project = None
            continue
        results = begin_dep.search(line)
        if results:
            in_deps = True
            continue
        results = end_dep.search(line)
        if results:
            in_deps = False
            continue
        results = dep_line.search(line)
        if results and in_deps and current_project:
            dependencies[current_project].append(results.group(1))
            continue
    for project in dependencies:
        new_dep_array = []
        for dep in dependencies[project]:
            for project_info in projects:
                if projects[project_info][1] == dep:
                    new_dep_array.append(project_info)
        dependencies[project] = sorted(new_dep_array)
    return (projects, dependencies)