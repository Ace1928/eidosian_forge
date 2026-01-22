from os.path import exists
import sys
from collections import defaultdict
import json

Changelog parser
================

This generates a changelog from a json file of the PRs of a given milestone,
dumped to json, using the [GitHub CLI](https://github.com/cli/cli).

First, in the command line, create the following alias::

    gh alias set --shell viewMilestone "gh api graphql -F owner='kivy' -F name='kivy' -F number=\$1 -f query='
        query GetMilestones(\$name: String!, \$owner: String!, \$number: Int!) {
            repository(owner: \$owner, name: \$name) {
                milestone(number: \$number) {
                    pullRequests(states: MERGED, first: 1000) {
                        nodes {
                            number
                            title
                            labels (first: 25) {
                                nodes {
                                    name
                                }
                            }
                        }
                    }
                }
            }
        }
    '"

Then, log in using ``gh`` and run::

    gh viewMilestone 26 > prs.json

This will generate ``prs.json``. Then, to generate the changelog, run::

    python -m kivy.tools.changelog_parser prs.json changelog.md

to generate a markdown changelog at ``changelog.md``. Then, edit as desired
and paste into the
[changelog here](https://github.com/kivy/kivy/blob/master/doc/sources/changelog.rst).
