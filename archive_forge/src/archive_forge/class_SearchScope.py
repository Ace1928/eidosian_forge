from enum import Enum, IntEnum
from gitlab._version import __title__, __version__
class SearchScope(GitlabEnum):
    PROJECTS: str = 'projects'
    ISSUES: str = 'issues'
    MERGE_REQUESTS: str = 'merge_requests'
    MILESTONES: str = 'milestones'
    WIKI_BLOBS: str = 'wiki_blobs'
    COMMITS: str = 'commits'
    BLOBS: str = 'blobs'
    USERS: str = 'users'
    GLOBAL_SNIPPET_TITLES: str = 'snippet_titles'
    PROJECT_NOTES: str = 'notes'