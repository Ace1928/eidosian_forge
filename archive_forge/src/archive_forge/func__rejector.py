import futurist
def _rejector(executor, backlog):
    if backlog >= max_backlog:
        raise futurist.RejectedSubmission('Current backlog %s is not allowed to go beyond %s' % (backlog, max_backlog))