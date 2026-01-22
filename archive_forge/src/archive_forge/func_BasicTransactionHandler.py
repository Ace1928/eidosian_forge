from paste.httpexceptions import HTTPException
from wsgilib import catch_errors
def BasicTransactionHandler(application, factory):
    """
    Provides a simple mechanism for starting a transaction based on the
    factory; and for either committing or rolling back the transaction
    depending on the result.  It checks for the response's current
    status code either through the latest call to start_response; or
    through a HTTPException's code.  If it is a 100, 200, or 300; the
    transaction is committed; otherwise it is rolled back.
    """

    def basic_transaction(environ, start_response):
        conn = factory(environ)
        environ['paste.connection'] = conn
        should_commit = [500]

        def finalizer(exc_info=None):
            if exc_info:
                if isinstance(exc_info[1], HTTPException):
                    should_commit.append(exc_info[1].code)
            if should_commit.pop() < 400:
                conn.commit()
            else:
                try:
                    conn.rollback()
                except:
                    return
            conn.close()

        def basictrans_start_response(status, headers, exc_info=None):
            should_commit.append(int(status.split(' ')[0]))
            return start_response(status, headers, exc_info)
        return catch_errors(application, environ, basictrans_start_response, finalizer, finalizer)
    return basic_transaction