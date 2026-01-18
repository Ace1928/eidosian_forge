from breezy import errors
from breezy.bzr import knit
from breezy.tests.per_repository_reference import \
intermix the revisions so that base holds left stacked holds right.

        base will hold
            A B D F (and C because it is a parent of D)
        referring will hold
            C E G (only)
        