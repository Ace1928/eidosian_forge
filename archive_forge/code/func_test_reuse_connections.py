from __future__ import print_function
import boto
import time
import uuid
from threading import Thread
def test_reuse_connections():
    """
    This test is an attempt to expose problems because of the fact
    that boto returns connections to the connection pool before
    reading the response.  The strategy is to start a couple big reads
    from S3, where it will take time to read the response, and then
    start other requests that will reuse the same connection from the
    pool while the big response is still being read.

    The test passes because of an interesting combination of factors.
    I was expecting that it would fail because two threads would be
    reading the same connection at the same time.  That doesn't happen
    because httplib catches the problem before it happens and raises
    an exception.

    Here's the sequence of events:

       - Thread 1: Send a request to read a big S3 object.
       - Thread 1: Returns connection to pool.
       - Thread 1: Start reading the body if the response.

       - Thread 2: Get the same connection from the pool.
       - Thread 2: Send another request on the same connection.
       - Thread 2: Try to read the response, but
                   HTTPConnection.get_response notices that the
                   previous response isn't done reading yet, and
                   raises a ResponseNotReady exception.
       - Thread 2: _mexe catches the exception, does not return the
                   connection to the pool, gets a new connection, and
                   retries.

       - Thread 1: Finish reading the body of its response.
       
       - Server:   Gets the second request on the connection, and
                   sends a response.  This response is ignored because
                   the connection has been dropped on the client end.

    If you add a print statement in HTTPConnection.get_response at the
    point where it raises ResponseNotReady, and then run this test,
    you can see that it's happening.
    """
    print('Running test_reuse_connections')
    s3 = boto.connect_s3()
    bucket = s3.create_bucket('test-%d' % int(time.time()))
    small_names = [str(uuid.uuid4()) for _ in range(4)]
    for i, name in enumerate(small_names):
        bucket.new_key(name).set_contents_from_string(str(i))
    print('    waiting for all connections to become stale')
    time.sleep(s3._pool.STALE_DURATION + 1)
    s3._pool.clean()
    assert s3._pool.size() == 0
    print('    pool is empty')
    big_name = str(uuid.uuid4())
    contents = '-' * BIG_SIZE
    bucket.new_key(big_name).set_contents_from_string(contents)
    threads = [spawn(read_big_object, s3, bucket, big_name, 20) for _ in range(5)]
    queriers = [LittleQuerier(bucket, small_names) for _ in range(5)]
    for t in threads:
        t.join()
    for q in queriers:
        q.stop()