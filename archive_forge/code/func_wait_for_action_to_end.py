import time
def wait_for_action_to_end(timeout, func, **kwargs):
    count = 0
    while count < timeout:
        if func(**kwargs):
            return True
        count += 1
        time.sleep(1)
    return False