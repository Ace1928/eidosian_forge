import time
import logging
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from dash.testing.errors import TestingTimeoutError
def until_not(wait_cond, timeout, poll=0.1, msg='expected condition met within timeout'):
    res = wait_cond()
    logger.debug('start wait.until_not method, timeout, poll => %s %s %s', wait_cond, timeout, poll)
    end_time = time.time() + timeout
    while res:
        if time.time() > end_time:
            raise TestingTimeoutError(msg)
        time.sleep(poll)
        res = wait_cond()
        logger.debug('poll => %s', time.time())
    return res