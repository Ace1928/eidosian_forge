import sys
 When iterating over b'...' in Python 2 you get single b'_' chars
            and in Python 3 you get integers. Call bchr to always turn this
            to single b'_' chars.
        